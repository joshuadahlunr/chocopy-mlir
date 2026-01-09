#include "AST.hpp"
#include "parser.hpp"
#include "string_helpers.hpp"
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_set>

namespace sema {

	struct resolve_lookups : public AST::visiter<AST::ref> {
		std::string_view source;
		bool as_type = false;

		template<typename Tfunction>
		auto as_type_block(const Tfunction& func) {
			auto back = as_type; as_type = true;
			if constexpr (std::is_same_v<decltype(func()), void>) {
				func();
				as_type = back;
			} else {
				auto out = func();
				as_type = back;
				return out;
			}
		}

		struct symbol_table : protected std::vector<std::unordered_map<interned_string, AST::ref>> {
			std::unordered_map<interned_string, AST::ref>& peek() { return back(); }

			void push_scope() {
				this->emplace_back();
			}

			void pop_scope() {
				this->pop_back();
			}

			std::optional<std::pair<AST::ref, size_t>> find(interned_string what) {
				for(size_t i = size(); i--; ) {
					auto& table = at(i);
					if(auto iter = table.find(what); iter != table.end())
						return {{iter->second, size() - i}};
				}
				return {};
			}

			size_t try_insert(interned_string what, AST::ref ref) {
				auto level = find(what);
				if(level && level->second == 0) return 0;

				peek()[what] = ref;
				return level ? level->second : AST::absent;
			}
		} symbol_table;

		explicit resolve_lookups(AST::flattened &ast, std::string_view source) : visiter<AST::ref>(ast), source(source) {}

		bool is_declaration(AST::ref r) {
			return ast[r].is_variable_declaration()
				|| ast[r].is_class_declaration()
				|| ast[r].is_class_declaration_lookup()
				|| ast[r].is_function_declaration()
				|| ast[r].is_parameter_declaration()
				|| ast[r].is_global()
				|| ast[r].is_global_lookup()
				|| ast[r].is_nonlocal()
				|| ast[r].is_nonlocal_lookup();
		}

		interned_string get_decl_name(AST::ref r) {
			auto& node = ast[r];
			switch (node.index()) {
			case AST::detail::variant_index_v<AST::variable_declaration, AST::node::variant>:
				return node.as_variable_declaration().name;
			case AST::detail::variant_index_v<AST::class_declaration, AST::node::variant>:
				return node.as_class_declaration().name;
			case AST::detail::variant_index_v<AST::class_declaration_lookup, AST::node::variant>:
				return node.as_class_declaration_lookup().name;
			case AST::detail::variant_index_v<AST::function_declaration, AST::node::variant>:
				return node.as_function_declaration().name;
			case AST::detail::variant_index_v<AST::parameter_declaration, AST::node::variant>:
				return node.as_parameter_declaration().name;
			case AST::detail::variant_index_v<AST::global, AST::node::variant>:
				return get_decl_name(node.as_global().reference);
			case AST::detail::variant_index_v<AST::global_lookup, AST::node::variant>:
				return node.as_global_lookup().interned_name;
			case AST::detail::variant_index_v<AST::nonlocal, AST::node::variant>:
				return get_decl_name(node.as_nonlocal().reference);
			case AST::detail::variant_index_v<AST::nonlocal_lookup, AST::node::variant>:
				return node.as_nonlocal_lookup().interned_name;
			default: throw std::runtime_error("Trying to get decl name for non-decl");
			}
		}

		std::vector<std::reference_wrapper<AST::block>> get_blocks(AST::ref r) {
			auto& node = ast[r];
			switch (node.index()) {
			case AST::detail::variant_index_v<AST::block, AST::node::variant>:
				return {node.as_block()};
			case AST::detail::variant_index_v<AST::class_declaration, AST::node::variant>:
				return {node.as_class_declaration()};
			case AST::detail::variant_index_v<AST::class_declaration_lookup, AST::node::variant>:
				return {node.as_class_declaration_lookup()};
			case AST::detail::variant_index_v<AST::function_declaration, AST::node::variant>:
				return {node.as_function_declaration()};
			case AST::detail::variant_index_v<AST::if_statement, AST::node::variant>: {
				std::vector<std::reference_wrapper<AST::block>> out;
				for(auto& cbp: node.as_if_statement().condition_block_pairs)
					out.push_back(cbp.block);
				return out;
			}
			case AST::detail::variant_index_v<AST::while_statement, AST::node::variant>:
				return {node.as_while_statement()};
			case AST::detail::variant_index_v<AST::for_statement, AST::node::variant>:
				return {node.as_for_statement()};
			default: throw std::runtime_error("Trying to get decl name for non-decl");
			}
		}

		std::optional<AST::ref> lookup(interned_string what, AST::ref start) {
			// Ask the symbol table first... that is much faster
			auto found = symbol_table.find(what);
			if(found) return found->first;

			// If the symbol table fails exhaustively search every block we are a child of
			AST::ref block = ast[start].as_node_base().scope_block;
			while(!found && block != AST::absent) {
				auto blocks = get_blocks(block);
				for(auto& b: blocks)
					for(auto r: b.get().elements)
						if(is_declaration(r) && get_decl_name(r) == what)
							return r;
				block = blocks.front().get().scope_block;
			}

			return {};
		}

		AST::ref get_global_scope_ref() {
			return ast[0].as_block().elements.back();
		}
		AST::block& get_global_scope() {
			return ast[get_global_scope_ref()].as_block();
		}


		AST::ref visit_ref(AST::ref& target, AST::ref r) override {
			target = visit(target);
			return r;
		}

		AST::ref visit_type_lookup(AST::type_lookup& value, AST::ref r) override {
			auto found = lookup(value.interned_name, r);
			if(!found) {
				diagnostics::singleton().push_error("Class has not be declared", source, value.location);
				return r;
			}
			if( !(ast[*found].is_class_declaration() || ast[*found].is_class_declaration_lookup()) ) {
				diagnostics::singleton().push_error("This appears to not be a type", source, value.location);
				return r;
			}

			return *found;
		}

		AST::ref visit_list_type(AST::list_type& value, AST::ref r) override {
			as_type_block([&](){ value.type = visit(value.type); });
			return r;
		}

		AST::ref visit_class_declaration_lookup(AST::class_declaration_lookup& value, AST::ref r) override {
			auto level = symbol_table.try_insert(value.name, r);
			if(level == 0)
				diagnostics::singleton().push_error("Attempting to redeclare class", source, value.location);
			if(level != AST::absent)
				diagnostics::singleton().push_warning("Class shadows another symbol", source, value.location);

			auto base = lookup(value.base, r);
			if(!base) {
				diagnostics::singleton().push_error("Base class has not been declared", source, value.location);
				return r;
			}

			// Resolve the lookup
			AST::class_declaration cls = {{{value.location, value.scope_block}}};
			cls.name = value.name;
			cls.elements = std::move(value.elements);
			cls.base = *base;
			ast[r] = {cls};

			return visit_block(cls, r);
		}

		AST::ref visit_class_declaration(AST::class_declaration& value, AST::ref r) override {
			if(as_type) return r;

			auto level = symbol_table.try_insert(value.name, r);
			if(level == 0)
				diagnostics::singleton().push_error("Attempting to redeclare class", source, value.location);
			if(level != AST::absent)
				diagnostics::singleton().push_warning("Class shadows another symbol", source, value.location);
			return visit_block(value, r);
		}

		AST::ref visit_function_declaration(AST::function_declaration& value, AST::ref r) override {
			auto level = symbol_table.try_insert(value.name, r);
			if(level == 0)
				diagnostics::singleton().push_error("Attempting to redeclare function", source, value.location);
			if(level != AST::absent)
				diagnostics::singleton().push_warning("Function shadows another symbol", source, value.location);
			return visit_block(value, r);
		}

		AST::ref visit_parameter_declaration(AST::parameter_declaration& value, AST::ref r) override {
			auto level = symbol_table.try_insert(value.name, r);
			if(level == 0)
				diagnostics::singleton().push_error("Attempting to redeclare parameter", source, value.location);
			// if(level != AST::absent) // TODO: crashes!
			// 	diagnostics::singleton().push_warning("Parameter shadows another symbol", source, value.location);
			if(value.type != AST::absent) as_type_block([&](){ value.type = visit(value.type); });
			return r;
		}

		AST::ref visit_variable_declaration(AST::variable_declaration& value, AST::ref r) override {
			auto level = symbol_table.try_insert(value.name, r);
			if(level == 0)
				diagnostics::singleton().push_error("Attempting to redeclare variable", source, value.location);
			if(level != AST::absent)
				diagnostics::singleton().push_warning("Variable shadows another symbol", source, value.location);
			as_type_block([&](){ value.type = visit(value.type); });
			value.initial_value = visit(value.initial_value);
			return r;
		}

		AST::ref visit_global_lookup(AST::global_lookup& value, AST::ref r) override {
			std::optional<AST::ref> found = {};
			for(auto ref: get_global_scope().elements)
				if(is_declaration(ref) && get_decl_name(ref) == value.interned_name) {
					found = ref;
					break;
				}
			if(!found) {
				diagnostics::singleton().push_error("Attempting to reference non-existant global variable", source, value.location);
				return r;
			}

			auto level = symbol_table.try_insert(value.interned_name, r);
			if(level == 0) {
				diagnostics::singleton().push_error("Attempting to redeclare global variable", source, value.location);
				return r;
			}
			// if(level != AST::absent)
			// 	diagnostics::singleton().push_warning("Global variable shadows another symbol", source, value.location);

			AST::global g = {{value.location, value.scope_block}};
			g.reference = *found;
			ast[r] = {g};

			return r;
		}

		AST::ref visit_global(AST::global& value, AST::ref r) override {
			if(!is_declaration(value.reference))
				diagnostics::singleton().push_error("Global does not point to a declaration", source, value.location);

			return r;
		}

		AST::ref visit_nonlocal_lookup(AST::nonlocal_lookup& value, AST::ref r) override {
			auto found = symbol_table.find(value.interned_name);
			if(!found) {
				diagnostics::singleton().push_error("Attempting to reference non-existant nonlocal variable", source, value.location);
				return r;
			}

			for(auto ref: get_global_scope().elements)
				if(ref == found->first) {
					diagnostics::singleton().push_error("Nonlocals can't be in the global scope", source, value.location);
					return r;
				}

			auto level = symbol_table.try_insert(value.interned_name, r);
			if(level == 0) {
				diagnostics::singleton().push_error("Attempting to redeclare nonlocal variable", source, value.location);
				return r;
			}
			// if(level != AST::absent)
			// 	diagnostics::singleton().push_warning("Nonlocal variable shadows another symbol", source, value.location);

			AST::nonlocal nl = {{value.location, value.scope_block}};
			nl.reference = found->first;
			ast[r] = {nl};

			return r;
		}

		AST::ref visit_nonlocal(AST::nonlocal& value, AST::ref r) override {
			if(!is_declaration(value.reference))
				diagnostics::singleton().push_error("Nonlocal does not point to a declaration", source, value.location);
			return r;
		}

		AST::ref visit_pass_statement(AST::pass_statement&, AST::ref r) override {
			return r;
		}

		AST::ref visit_return_statement(AST::return_statement& value, AST::ref r) override {
			value.what = visit(value.what);
			return r;
		}

		AST::ref visit_assignment(AST::assignment& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_if_statement(AST::if_statement& value, AST::ref r) override {
			for(auto& [condition, block]: value.condition_block_pairs) {
				if(condition != AST::absent) condition = visit(condition);
				visit_block(block, r);
			}
			return r;
		}

		AST::ref visit_while_statement(AST::while_statement& value, AST::ref r) override {
			value.condition = visit(value.condition);
			return visit_block(value, r);
		}

		AST::ref visit_for_statement(AST::for_statement& value, AST::ref r) override {
			value.source = visit(value.source);
			return visit_block(value, r);
		}

		AST::ref visit_block(AST::block& value, AST::ref r) override {
			symbol_table.push_scope();
			for (auto &elem : value.elements)
				elem = visit(elem);
			symbol_table.pop_scope();
			return r;
		}

		AST::ref visit_if_expression(AST::if_expression& value, AST::ref r) override {
			value.then = visit(value.then);
			value.condition = visit(value.condition);
			value.else_ = visit(value.else_);
			return r;
		}

		AST::ref visit_logical_and(AST::logical_and& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_logical_or(AST::logical_or& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_equal(AST::equal& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_not_equal(AST::not_equal& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_less(AST::less& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_less_equal(AST::less_equal& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_greater(AST::greater& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_greater_equal(AST::greater_equal& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_is(AST::is& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_add(AST::add& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_subtract(AST::subtract& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_multiply(AST::multiply& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_quotient(AST::quotient& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_remainder(AST::remainder& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_divide(AST::divide& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_logical_not(AST::logical_not& value, AST::ref r) override {
			value.what = visit(value.what);
			return r;
		}

		AST::ref visit_negate(AST::negate& value, AST::ref r) override {
			value.what = visit(value.what);
			return r;
		}

		AST::ref visit_variable_load_lookup(AST::variable_load_lookup& value, AST::ref r) override {
			auto found = lookup(value.interned_name, r);
			if(!found) {
				diagnostics::singleton().push_error("Attempting to access non-existant variable", source, value.location);
				return r;
			}

			AST::variable_load vs = {value.location, value.scope_block};
			vs.reference = *found;
			vs.type = value.type;
			ast[r] = {vs};

			return r;
		}

		AST::ref visit_variable_load(AST::variable_load& value, AST::ref r) override {
			if(value.reference == AST::absent) 
				diagnostics::singleton().push_error("Attempting to access non-existant variable", source, value.location);
			return r;
		}

		AST::ref visit_variable_store_lookup(AST::variable_store_lookup& value, AST::ref r) override {
			// Stores can't be functions so we just need to do a symbol table search
			auto found = symbol_table.find(value.interned_name);
			if(!found) {
				diagnostics::singleton().push_error("Attempting to assign to non-existant variable", source, value.location);
				return r;
			}
			
			if(ast[found->first].is_function_declaration()) {
				diagnostics::singleton().push_error("Attempting to assign to a function", source, value.location);
				return r;
			}
			if(ast[found->first].is_class_declaration() || ast[found->first].is_class_declaration_lookup()) {
				diagnostics::singleton().push_error("Attempting to assign to a class", source, value.location);
				return r;
			}

			AST::variable_store vs = {value.location, value.scope_block};
			vs.reference = found->first;
			vs.type = value.type;
			ast[r] = {vs};

			return r;
		}

		AST::ref visit_variable_store(AST::variable_store& value, AST::ref r) override {
			if(value.reference == AST::absent) 
				throw std::runtime_error("Invalid variable store");

			if(ast[value.reference].is_function_declaration()) {
				diagnostics::singleton().push_error("Attempting to assign to a function", source, value.location);
				return r;
			}
			if(ast[value.reference].is_class_declaration() || ast[value.reference].is_class_declaration_lookup()) {
				diagnostics::singleton().push_error("Attempting to assign to a class", source, value.location);
				return r;
			}

			return r;
		}

		AST::ref visit_member_access_lookup(AST::member_access_lookup& value, AST::ref r) override {
			if(value.type == AST::absent) { // Member accesses are dependent on type information!
				visit(value.lhs);
				return r;
			}

			// TODO:: Lookup
			return r;
		}

		AST::ref visit_member_access(AST::member_access& expr, AST::ref) override {
			// return visit(expr.lhs) + ".lookup(" + std::string(expr.interned_name) + ")";
			throw std::runtime_error("Not implemented yet!");
		}

		AST::ref visit_array_index(AST::array_index& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			value.rhs = visit(value.rhs);
			return r;
		}

		AST::ref visit_call(AST::call& value, AST::ref r) override {
			value.lhs = visit(value.lhs);
			for(auto& arg: value.elements)
				arg = visit(arg);
			return r;
		}

		AST::ref visit_float_literal(AST::float_literal& value, AST::ref r) override {
			return r;
		}

		AST::ref visit_int_literal(AST::int_literal& value, AST::ref r) override {
			return r;
		}

		AST::ref visit_string_literal(AST::string_literal& value, AST::ref r) override {
			return r;
		}

		AST::ref visit_bool_literal(AST::bool_literal& value, AST::ref r) override {
			return r;
		}

		AST::ref visit_none_literal(AST::none_literal& value, AST::ref r) override {
			return r;
		}

		AST::ref visit_list_literal(AST::list_literal& value, AST::ref r) override {
			for(auto& elem: value.elements)
				elem = visit(elem);
			return r;
		}
	};

} // namespace AST