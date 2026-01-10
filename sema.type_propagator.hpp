#include "AST.hpp"
#include "AST.print.hpp"
#include "string_helpers.hpp"
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sema {

	struct type_propagator : public AST::visiter<AST::ref> {
		string_interner& interner;
		std::string_view source;
		size_t builtin_size;

		explicit type_propagator(AST::flattened &ast, string_interner& interner, std::string_view source, size_t builtin_size) 
			: visiter<ref>(ast), interner(interner), source(source), builtin_size(builtin_size) {}

		using ref = AST::ref;
		const ref absent = AST::absent;

		ref none = absent;
		ref empty = absent;
		ref Int = absent;
		ref Float = absent;
		ref Bool = absent;
		ref Str = absent;

		bool builtin_types_missing() {
			return none == absent
				|| empty == absent
				|| Int == absent
				|| Float == absent
				|| Bool == absent
				|| Str == absent;
		}

		bool type_equivalent(ref a_ref, ref b_ref) {
			if(a_ref == absent || b_ref == absent)
				return false;

			AST::node& a = ast[a_ref], &b = ast[b_ref];
			if(a.index() != b.index())
				return false;

			if(a.is_class_declaration())
				return a_ref == b_ref;

			if(a.is_list_type())
				return type_equivalent(a.as_list_type().type, b.as_list_type().type);

			if(a.is_function_type()) {
				auto& af = a.as_function_type(), &bf = b.as_function_type();
				if(!type_equivalent(af.return_type, bf.return_type))
					return false;
				if(af.elements.size() != bf.elements.size())
					return false;
				for(size_t i = 0; i < af.elements.size(); ++i)
					if(!type_equivalent(af.elements[i], bf.elements[i]))
						return false;
				return true;
			}

			return false;
		}

		bool type_convertible(ref from_ref, ref to_ref) {
			if(from_ref == absent || to_ref == absent)
				return false;
			if(type_equivalent(from_ref, to_ref))
				return true;
			auto& from = ast[from_ref], &to = ast[to_ref];

			// T1 ≤ T2 (i.e., ordinary subtyping).
			if(from.is_class_declaration() && to.is_class_declaration())
				return type_convertible(from.as_class_declaration().base, to_ref);

			// from = int | float, to = int | float
			// if((from_ref == Int || from_ref == Float) && (to_ref == Int || to_ref == Float))
			// 	return true; // Unnecessary since they are now subtypes?

			// T1 is <None> and T2 is not int, bool, or str.
			if(from_ref == none) {
				if(to_ref == Int || to_ref == Float || to_ref == Bool || to_ref == Str)
					return false;
				return true;
			}

			// T2 is a list type [T] and T1 is <Empty>.
			if(from_ref == empty && to.is_list_type())
				return true;

			//  T2 is a the list type [T] and T1 is [<None>], where <None> ≤a T.
			if(from.is_list_type() && from.as_list_type().type == none && to.is_list_type())
				return true;

			return false;
		}

		ref greatest_type(ref a, ref b) {
			// NOTE: Since int is a subclass of float this will allow ints to convert to floats
			if(type_convertible(a, b))
				return b;
			if(type_convertible(b, a))
				return a;
			return absent;
		}

		ref check_expression(ref& lhs, ref& rhs, const diagnostics::source_location& location) {
			auto lhs_type = visit(lhs);
			auto rhs_type = visit(rhs);
			if(lhs_type == absent || rhs_type == absent) return absent;
			if( !(type_convertible(lhs_type, rhs_type) || type_convertible(rhs_type, lhs_type)) ) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "The two sides of this expression have incompatible types `" + p.visit(lhs_type) + "` and `" + p.visit(rhs_type) + "`";
				diagnostics::singleton().push_error(hint, source, location);
				return none;
			}

			if(!type_equivalent(lhs_type, rhs_type)) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto greatest = greatest_type(lhs_type, rhs_type);
				auto& fix = lhs_type == greatest ? rhs : lhs;
				auto& fix_type = lhs_type == greatest ? rhs_type : lhs_type;
				auto& fix_node_base = ast[fix].as_node_base();
				
				auto hint = "Inserted implicit conversion from `" + p.visit(lhs_type) + "` to `" + p.visit(greatest) + "`";
				diagnostics::singleton().push_warning(hint, source, fix_node_base.location);
					
				AST::explicit_cast cast = {fix_node_base};
				cast.reference = fix;
				cast.type = greatest;
				fix = AST::make_node(ast, cast);	
				return greatest;
			}

			return lhs_type;
		}

		ref check_expression_expected(ref& r, const std::span<const ref> expected_types, const diagnostics::source_location& location, std::optional<ref> type_override = {}) {
			assert(!expected_types.empty());
			auto type = type_override ? *type_override : visit(r);
			if(type == absent) return absent;

			std::optional<ref> valid_type = {};
			for(auto expected: expected_types)
				if(type_convertible(type, expected)) {
					valid_type = expected;
					break;
				}
			if(!valid_type) {
				AST::pretty_printer p(ast); p.name_only = true;
				std::string hint = "Expected ";
				if(expected_types.size() > 1)
					hint += "one of ";
				for(auto type: expected_types)
					hint += "`" + p.visit(type) + "` ";
				hint += "received `" + p.visit(type) + "`";
				diagnostics::singleton().push_error(hint, source, location);
				return none;	
			}

			if(!type_equivalent(type, *valid_type)) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Inserted implicit conversion from `" + p.visit(type) + "` to `" + p.visit(*valid_type) + "`";
				diagnostics::singleton().push_warning(hint, source, location);
				
				AST::explicit_cast cast = {ast[r].as_node_base()};
				cast.reference = r;
				cast.type = *valid_type;
				r = AST::make_node(ast, cast);
			}

			return *valid_type;
		}

		bool is_maybe_inside_function(ref scope) {
			return ast[scope].is_function_declaration()
				|| ast[scope].is_if_statement()
				|| ast[scope].is_while_statement()
				|| ast[scope].is_for_statement();
		}

		std::unordered_map<ref, ref> memoization_cache; // maps r -> type
		std::unordered_map<ref, ref> list_type_cache; // maps inside -> outside



		ref visit_ref(ref& target, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = visit(target);
		}

		ref visit_type_lookup(AST::type_lookup&, ref) override {
			throw std::runtime_error("All types should already be looked up!");
		}

		ref visit_list_type(AST::list_type& value, ref r) override {
			if(list_type_cache.contains(value.type)) return list_type_cache[value.type];
			return list_type_cache[value.type] = r;
		}

		ref visit_function_type(AST::function_type&, ref r) override {
			return r;
		}

		ref visit_class_declaration_lookup(AST::class_declaration_lookup& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			visit_block(value, r);
			return memoization_cache[r] = r;
		}

		ref visit_class_declaration(AST::class_declaration& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			visit_block(value, r);
			return memoization_cache[r] = r;
		}

		ref visit_function_declaration(AST::function_declaration& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			visit_block(value, r);

			// static std::unordered_map<ref, ref> cache;
			// if(cache.find(r) != cache.end())
			// 	return cache[r];

			if(value.name == interner.intern("__init__")) {
				if( !(ast[value.scope_block].is_class_declaration() || ast[value.scope_block].is_class_declaration_lookup()) )
					diagnostics::singleton().push_error("Constructors can only appear inside the top level of a class", source, value.location);

				if(value.return_type == absent)
					value.return_type = value.scope_block; // No return -> return the class
				if(value.return_type != value.scope_block)
					diagnostics::singleton().push_error("Constructors must return their class type", source, value.location);
			}

			AST::function_type type;
			type.return_type = value.return_type == absent ? none : value.return_type;
			type.elements.resize(value.num_parameters);
			for(size_t i = 0; i < value.num_parameters; ++i)
				if(ast[value.elements[i]].is_parameter_declaration())
					type.elements[i] = ast[value.elements[i]].as_parameter_declaration().type;
				else type.elements[i] = absent;

			return memoization_cache[r] = AST::make_node(ast, type);
		}

		ref visit_parameter_declaration(AST::parameter_declaration& value, ref r) override {
			return value.type;
		}

		ref visit_variable_declaration(AST::variable_declaration& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			if(r > builtin_size) 
				check_expression_expected(value.initial_value, std::array<ref, 2>{value.type, none}, value.location);
			else visit(value.initial_value);
			return memoization_cache[r] = value.type;
		}

		ref visit_global_lookup(AST::global_lookup& value, ref r) override {
			return absent;
		}

		ref visit_global(AST::global& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = visit(value.reference);
		}

		ref visit_nonlocal_lookup(AST::nonlocal_lookup& value, ref r) override {
			return absent;
		}

		ref visit_nonlocal(AST::nonlocal& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = visit(value.reference);
		}

		ref visit_pass_statement(AST::pass_statement&, ref r) override {
			return none;
		}

		ref visit_return_statement(AST::return_statement& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto scope = value.scope_block;
			auto last = scope;
			while(scope != absent && is_maybe_inside_function(scope))
				last = scope, scope = ast[scope].as_node_base().scope_block;
			if(!ast[last].is_function_declaration()) {
				diagnostics::singleton().push_error("Return statements can only be inside of functions", source, value.location);
				return memoization_cache[r] = none;	
			}
			if(value.what != absent) {
				auto return_type = ast[last].as_function_declaration().return_type;
				if(return_type == absent) {
					diagnostics::singleton().push_error("Returning a value from a function with no return type", source, value.location);
					return memoization_cache[r] = none;
				}
				check_expression_expected(value.what, std::array<ref, 1>{return_type}, value.location);
			}

			return memoization_cache[r] = none;
		}

		ref visit_assignment(AST::assignment& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto lhs_type = visit(value.lhs);
			auto rhs_type = visit(value.rhs);

			if(lhs_type == absent)
				ast[value.lhs].as_expression().type = lhs_type = rhs_type;
			else if(!type_convertible(rhs_type, lhs_type)) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Failed to assign value of type `" + p.visit(rhs_type) + "` to variable of type `" + p.visit(lhs_type) + "`";
				diagnostics::singleton().push_error(hint, source, value.location);
				return memoization_cache[r] = value.type = lhs_type;	
			}

			if(auto type = check_expression_expected(value.rhs, std::array<ref, 1>{lhs_type}, value.location); type != none)
				return memoization_cache[r] = value.type = type;
			return memoization_cache[r] = value.type = lhs_type;
		}

		ref visit_if_statement(AST::if_statement& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			for(auto& [condition, block]: value.condition_block_pairs) {
				if(condition != absent)
					if(auto condition_type = visit(condition); condition_type != Bool) {
						diagnostics::singleton().push_error("If conditions must have type `bool`", source, ast[condition].as_node_base().location);
						return memoization_cache[r] = none;
					}
				visit_block(block, r);
			}
			return memoization_cache[r] = none;
		}

		ref visit_while_statement(AST::while_statement& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			if(value.condition != absent)
				if(auto condition_type = visit(value.condition); condition_type != Bool) {
					diagnostics::singleton().push_error("While conditions must have type `bool`", source, ast[value.condition].as_node_base().location);
					return memoization_cache[r] = none;
				}
			visit_block(value, r);
			return memoization_cache[r] = none;
		}

		ref visit_for_statement_lookup(AST::for_statement_lookup& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto source_type = visit(value.source);
			if( !(source_type == Str || ast[source_type].is_list_type()) ) {
				diagnostics::singleton().push_error("Fors can only iterate over `list`s and `str`s", source, ast[value.source].as_node_base().location);
				return memoization_cache[r] = none;
			}

			visit_block(value, r);
			return memoization_cache[r] = none;
		}

		ref visit_for_statement(AST::for_statement& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto source_type = visit(value.source);
			if( !(source_type == Str || ast[source_type].is_list_type()) ) {
				diagnostics::singleton().push_error("Fors can only iterate over `list`s and `str`s", source, ast[value.source].as_node_base().location);
				return memoization_cache[r] = none;
			}

			auto param_type = source_type == Str ? Str : ast[source_type].as_list_type().type;
			check_expression_expected(value.reference, std::array<ref, 1>{param_type}, value.location);

			visit_block(value, r);
			return memoization_cache[r] = none;
		}

		ref visit_block(AST::block& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			if(builtin_types_missing())
				for(auto elem: value.elements)
					if(ast[elem].is_class_declaration()) {
						if(std::string_view{ast[elem].as_class_declaration().name} == "<NONE>")
							none = elem;
						else if(std::string_view{ast[elem].as_class_declaration().name} == "<EMPTY>")
							empty = elem;
						else if(std::string_view{ast[elem].as_class_declaration().name} == "int")
							Int = elem;
						else if(std::string_view{ast[elem].as_class_declaration().name} == "float")
							Float = elem;
						else if(std::string_view{ast[elem].as_class_declaration().name} == "bool")
							Bool = elem;
						else if(std::string_view{ast[elem].as_class_declaration().name} == "str")
							Str = elem;
					}

			for(auto elem: value.elements)
				visit(elem);
			return memoization_cache[r] = none;
		}



		ref visit_if_expression(AST::if_expression& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			if(auto condition_type = visit(value.condition); !(condition_type == Bool || condition_type == absent)) {
				diagnostics::singleton().push_error("If conditions must have type `bool`", source, ast[value.condition].as_node_base().location);
				return memoization_cache[r] = value.type = absent;
			}

			return memoization_cache[r] = value.type = check_expression(value.then, value.else_, value.location);
		}

		ref visit_explicit_cast(AST::explicit_cast& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			visit(value.reference);
			return memoization_cache[r] = value.type;
		}

		ref visit_logical_and(AST::logical_and& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, std::array<ref, 1>{Bool}, value.location, type) != none)
				check_expression_expected(value.rhs, std::array<ref, 1>{Bool}, value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_logical_or(AST::logical_or& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, std::array<ref, 1>{Bool}, value.location, type) != none)
				check_expression_expected(value.rhs, std::array<ref, 1>{Bool}, value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		std::vector<ref> relational_allowed(bool equality_expression = false) {
			if(equality_expression)
				return {Int, Float, Str, Bool};
			return {Int, Float};
		}

		ref visit_equal(AST::equal& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, relational_allowed(true), value.location, type) != none)
				check_expression_expected(value.rhs, relational_allowed(true), value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_not_equal(AST::not_equal& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, relational_allowed(true), value.location, type) != none)
				check_expression_expected(value.rhs, relational_allowed(true), value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_less(AST::less& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, relational_allowed(), value.location, type) != none)
				check_expression_expected(value.rhs, relational_allowed(), value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_less_equal(AST::less_equal& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, relational_allowed(), value.location, type) != none)
				check_expression_expected(value.rhs, relational_allowed(), value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_greater(AST::greater& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, relational_allowed(), value.location, type) != none)
				check_expression_expected(value.rhs, relational_allowed(), value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_greater_equal(AST::greater_equal& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, relational_allowed(), value.location, type) != none)
				check_expression_expected(value.rhs, relational_allowed(), value.location, type);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_is(AST::is& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			// TODO: Should we limit the allowed input types?
			check_expression(value.lhs, value.rhs, value.location);
			return memoization_cache[r] = value.type = Bool;
		}

		ref visit_add(AST::add& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			std::array<ref, 4> add_allowed = {Int, Float, Str, empty};
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(type == absent) return memoization_cache[r] = absent;
			if( !(ast[type].is_list_type() || std::find(add_allowed.begin(), add_allowed.end(), type) != add_allowed.end()) ) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Add expressions only supports values of type `int`, `float`, `str`, and `list` received `" + p.visit(type) + "`";
				diagnostics::singleton().push_error(hint, source, value.location);
			}
			return memoization_cache[r] = value.type = type;
		}

		std::array<ref, 2> arithmetic_allowed() {
			return {Int, Float};
		}

		ref visit_subtract(AST::subtract& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, arithmetic_allowed(), value.location, type) != none)
				check_expression_expected(value.rhs, arithmetic_allowed(), value.location, type);
			return memoization_cache[r] = value.type = type;
		}

		ref visit_multiply(AST::multiply& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, arithmetic_allowed(), value.location, type) != none)
				check_expression_expected(value.rhs, arithmetic_allowed(), value.location, type);
			return memoization_cache[r] = value.type = type;
		}

		ref visit_quotient(AST::quotient& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, std::array<ref, 1>{Int}, value.location, type) != none)
				check_expression_expected(value.rhs, std::array<ref, 1>{Int}, value.location, type);
			return memoization_cache[r] = value.type = type;
		}

		ref visit_remainder(AST::remainder& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, std::array<ref, 1>{Int}, value.location, type) != none)
				check_expression_expected(value.rhs, std::array<ref, 1>{Int}, value.location, type);
			return memoization_cache[r] = value.type = type;
		}

		ref visit_divide(AST::divide& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto type = check_expression(value.lhs, value.rhs, value.location);
			if(check_expression_expected(value.lhs, std::array<ref, 1>{Float}, value.location, type) != none)
				check_expression_expected(value.rhs, std::array<ref, 1>{Float}, value.location, type);
			return memoization_cache[r] = value.type = type;
		}

		ref visit_logical_not(AST::logical_not& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = value.type = check_expression_expected(value.what, std::array<ref, 1>{Bool}, value.location);
		}

		ref visit_negate(AST::negate& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = value.type = check_expression_expected(value.what, arithmetic_allowed(), value.location);
		}

		ref visit_variable_load_lookup(AST::variable_load_lookup& value, ref r) override {
			return value.type = absent;
		}

		ref visit_variable_load(AST::variable_load& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = value.type = visit(value.reference);
		}

		ref visit_variable_store_lookup(AST::variable_store_lookup& value, ref r) override {
			return value.type = absent;
		}

		ref visit_variable_store(AST::variable_store& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			return memoization_cache[r] = value.type = visit(value.reference);
		}

		ref visit_member_access_lookup(AST::member_access_lookup& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto lhs_type = visit(value.lhs);
			if(lhs_type == absent) return memoization_cache[r] = absent;
			auto& lhs_type_node = ast[lhs_type];
			if( !(lhs_type_node.is_class_declaration() || lhs_type_node.is_class_declaration_lookup()) ) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Only classes have members, not `" + p.visit(lhs_type) + "`";
				diagnostics::singleton().push_error(hint, source, value.location);
				return memoization_cache[r] = value.type = absent;
			}
			value.resolve_type = lhs_type;
			return memoization_cache[r] = value.type = absent; // Tell the resolver what type to search through
		}

		ref visit_member_access(AST::member_access& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto lhs_type = visit(value.lhs);
			if(lhs_type == absent) return memoization_cache[r] = absent;
			auto& lhs_type_node = ast[lhs_type];
			if( !(lhs_type_node.is_class_declaration() || lhs_type_node.is_class_declaration_lookup()) ) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Only classes have members, not `" + p.visit(lhs_type) + "`";
				diagnostics::singleton().push_error(hint, source, value.location);
				return memoization_cache[r] = value.type = absent;
			}
			if(!lhs_type_node.is_class_declaration()) return memoization_cache[r] = value.type = absent;
			
			auto base = lhs_type;
			bool found = false;
			while(base != absent) {
				auto& cls = ast[base].as_class_declaration();
				if(std::find(cls.elements.begin(), cls.elements.end(), value.reference) != cls.elements.end()) {
					found = true;
					break;;
				}
				base = cls.base;
			}
			if(!found)
				throw std::runtime_error("Invalid member state");

			return memoization_cache[r] = value.type = visit(value.reference);
		}

		ref visit_array_index(AST::array_index& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto lhs_type = visit(value.lhs);
			if(lhs_type == absent) return memoization_cache[r] = absent;
			if( !(lhs_type == Str || ast[lhs_type].is_list_type()) ) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Only `list`s and `str`s can be indexed, not `" + p.visit(lhs_type) + "`";
				diagnostics::singleton().push_error(hint, source, value.location);
				return memoization_cache[r] = value.type = absent;
			}

			return memoization_cache[r] = value.type = lhs_type == Str ? Str : ast[lhs_type].as_list_type().type;
		}

		ref visit_call(AST::call& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			auto lhs_type = visit(value.lhs);
			if(lhs_type == absent) return memoization_cache[r] = absent;
			auto& lhs_type_node = ast[lhs_type];
			if( !(lhs_type_node.is_function_type() || lhs_type_node.is_class_declaration() || lhs_type_node.is_class_declaration_lookup()) ) {
				AST::pretty_printer p(ast); p.name_only = true;
				auto hint = "Only functions and classes can be called, not `" + p.visit(lhs_type) + "`";
				diagnostics::singleton().push_error(hint, source, value.location);
				return memoization_cache[r] = value.type = absent;
			}

			auto reference = std::visit([this](auto& node) {
				if constexpr (requires (decltype(node) node) { node.reference; })
					return node.reference;
				return absent;
			}, ast[value.lhs]);
			if(reference == absent) 
				return memoization_cache[r] = absent;
			bool is_constructor = ast[reference].is_class_declaration() || ast[reference].is_class_declaration_lookup();
			bool is_method = !is_constructor && (ast[value.lhs].is_member_access() || ast[value.lhs].is_member_access_lookup());

			if(is_constructor && ast[reference].is_class_declaration()) {
				auto __init__ = interner.intern("__init__");
				bool found = false;
				auto& cls = ast[reference].as_class_declaration();
				for(auto mem: cls.elements)
					if(ast[mem].is_function_declaration() && ast[mem].as_function_declaration().name == __init__) {
						reference = mem;
						found = true;
						break;
					}

				// If a class doesn't have a constructor we make a default one!
				if(!found) {
					AST::function_declaration decl = {cls.location, reference};
					decl.name = __init__;
					decl.num_parameters = 0;
					decl.return_type = reference;
					cls.elements.push_back(reference = AST::make_node(ast, decl));
					ast.back().as_function_declaration().elements.push_back(AST::make_node(ast, AST::pass_statement{cls.location, reference}));
				}
			}

			if(!ast[reference].is_function_declaration()) {
				diagnostics::singleton().push_error("NOTE: Tried to call this", source, ast[reference].as_node_base().location);
				diagnostics::singleton().push_error("Only valid functions can be called", source, value.location);
				return memoization_cache[r] = value.type = absent;
			}

			auto func_type_ref = visit(reference);
			auto func_type = ast[func_type_ref].as_function_type();
			// If the first element of a method is the class itself, remove it from arity consideration
			if(is_method) {
				auto cls_ref = visit(ast[value.lhs].as_member_access().lhs);
				if(type_convertible(cls_ref, func_type.elements.front()))
					func_type.elements.erase(func_type.elements.begin());
			}

			if(func_type.elements.size() != value.elements.size()) {
				auto hint = "Attempted to call function which expects " + std::to_string(func_type.elements.size())
					+ " parameter(s) with " + std::to_string(value.elements.size()) + " argument(s)";
				diagnostics::singleton().push_error(hint, source, value.location);
				return memoization_cache[r] = value.type = absent;
			}

			for(size_t i = 0; i < func_type.elements.size(); ++i) {
				ref& arg = value.elements[i];
				if(func_type.elements[i] != absent)
					check_expression_expected(arg, std::array<ref, 1>{func_type.elements[i]}, ast[arg].as_node_base().location);
				else visit(arg);
			}

			return memoization_cache[r] = value.type = func_type.return_type;
		}

		ref visit_float_literal(AST::float_literal& value, ref r) override {
			return value.type = Float;
		}

		ref visit_int_literal(AST::int_literal& value, ref r) override {
			return value.type = Int;
		}

		ref visit_string_literal(AST::string_literal& value, ref r) override {
			return value.type = Str;
		}

		ref visit_bool_literal(AST::bool_literal& value, ref r) override {
			return value.type = Bool;
		}

		ref visit_none_literal(AST::none_literal& value, ref r) override {
			return value.type = none;
		}

		ref visit_list_literal(AST::list_literal& value, ref r) override {
			if(memoization_cache.contains(r)) return memoization_cache[r];
			std::vector<ref> element_types; element_types.reserve(value.elements.size());
			auto type = element_types.emplace_back(visit(value.elements.front()));
			for(size_t i = 1; i < value.elements.size(); ++i) {
				type = greatest_type(type, element_types.emplace_back(visit(value.elements[i])));
				if(type == absent) break;
			}
			if(type == absent) return memoization_cache[r] = value.type = absent;

			for(size_t i = 0; i < value.elements.size(); ++i)
				check_expression_expected(value.elements[i], std::array<ref, 1>{type}, value.location, element_types[i]);
			
			if(list_type_cache.contains(type)) return memoization_cache[r] = value.type = list_type_cache[type];

			AST::list_type list = {value.location, value.scope_block};
			list.type = type;
			return memoization_cache[r] = value.type = list_type_cache[type] = AST::make_node(ast, list);
		}
	};

} // namespace AST