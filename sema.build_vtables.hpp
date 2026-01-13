#include "AST.hpp"
#include "builtin_scope.hpp"
#include "string_helpers.hpp"

namespace sema {

	struct build_vtables : public AST::visiter<void> {
		string_interner& interner;
		std::string_view source;
		builtin_scope builtin;

		explicit build_vtables(AST::flattened &ast, string_interner& interner, std::string_view source, builtin_scope builtin) 
			: visiter<void>(ast), interner(interner), source(source), builtin(builtin) {}

		using ref = AST::ref;
		const ref absent = AST::absent;


		void visit_ref(ref& target, ref) override {
			visit(target);
		}

		void visit_type_lookup(AST::type_lookup&, ref) override {
			throw std::runtime_error("All types should already be looked up!");
		}

		void visit_list_type(AST::list_type&, ref) override {
			// Do nothing
		}

		void visit_function_type(AST::function_type&, ref) override {
			// Do nothing
		}

		void visit_class_declaration_lookup(AST::class_declaration_lookup&, ref) override {
			throw std::runtime_error("All classes should already be looked up!");
		}

		void visit_class_declaration(AST::class_declaration& value, ref r) override {
			visit_block(value, r);
		}

		void visit_function_declaration(AST::function_declaration& value, ref r) override {
			visit_block(value, r);

			if(ast[value.scope_block].is_class_declaration()) {
				std::stack<ref> bases;
				auto base = value.scope_block;
				while(base != absent) {
					bases.push(base);
					base = ast[base].as_class_declaration().base;
				}

				while(!bases.empty() && value.overloads == absent) {
					auto base = bases.top();
					bases.pop();

					auto& cls = ast[base].as_class_declaration();
					for(auto elem: cls.elements)
						if(ast[elem].is_function_declaration())
							if(ast[elem].as_function_declaration().name == value.name) {
								value.overloads = elem;
								break;
							}
				}

				// If we belong to someone else's overload set add us to it... 
				if(value.overloads)  
					ast[value.overloads].as_function_declaration().overloaded_by[value.scope_block] = r;
				// otherwise indicate that we belong to our own overload set
				else value.overloaded_by[value.scope_block] = r;
			}
		}

		void visit_parameter_declaration(AST::parameter_declaration&, ref) override {
			// Do nothing
		}

		void visit_variable_declaration(AST::variable_declaration& value, ref) override {
			visit(value.initial_value);
		}

		void visit_global_lookup(AST::global_lookup&, ref) override {
			// Do nothing
		}

		void visit_global(AST::global& value, ref) override {
			visit(value.reference);
		}

		void visit_nonlocal_lookup(AST::nonlocal_lookup&, ref) override {
			// Do nothing
		}

		void visit_nonlocal(AST::nonlocal& value, ref) override {
			visit(value.reference);
		}

		void visit_pass_statement(AST::pass_statement&, ref) override {
			// Do nothing
		}

		void visit_return_statement(AST::return_statement& value, ref) override {
			if(value.what != absent)
				visit(value.what);
		}

		void visit_assignment(AST::assignment& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_if_statement(AST::if_statement& value, ref r) override {
			for(auto& [condition, block]: value.condition_block_pairs) {
				visit(condition);
				visit_block(block, r);
			}
		}

		void visit_while_statement(AST::while_statement& value, ref r) override {
			if(value.condition != absent)
				visit(value.condition);
			visit_block(value, r);
		}

		void visit_for_statement_lookup(AST::for_statement_lookup& value, ref r) override {
			visit(value.source);
			visit_block(value, r);
		}

		void visit_for_statement(AST::for_statement& value, ref r) override {
			visit(value.source);
			visit(value.reference);
			visit_block(value, r);
		}

		void visit_block(AST::block& value, ref) override {
			for(auto elem: value.elements)
				visit(elem);
		}



		void visit_if_expression(AST::if_expression& value, ref) override {
			visit(value.then);
			visit(value.else_);
		}

		void visit_explicit_cast(AST::explicit_cast& value, ref) override {
			visit(value.reference);
		}

		void visit_logical_and(AST::logical_and& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_logical_or(AST::logical_or& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_equal(AST::equal& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_not_equal(AST::not_equal& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_less(AST::less& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_less_equal(AST::less_equal& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_greater(AST::greater& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_greater_equal(AST::greater_equal& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_is(AST::is& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_add(AST::add& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_subtract(AST::subtract& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_multiply(AST::multiply& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_quotient(AST::quotient& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_remainder(AST::remainder& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_divide(AST::divide& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_logical_not(AST::logical_not& value, ref) override {
			visit(value.what);
		}

		void visit_negate(AST::negate& value, ref) override {
			visit(value.what);
		}

		void visit_variable_load_lookup(AST::variable_load_lookup&, ref) override {
			// Do nothing
		}

		void visit_variable_load(AST::variable_load& value, ref) override {
			visit(value.reference);
		}

		void visit_variable_store_lookup(AST::variable_store_lookup&, ref) override {
			// Do nothing
		}

		void visit_variable_store(AST::variable_store& value, ref) override {
			visit(value.reference);
		}

		void visit_member_access_lookup(AST::member_access_lookup& value, ref) override {
			visit(value.lhs);
		}

		void visit_member_access(AST::member_access& value, ref) override {
			visit(value.lhs);
			visit(value.reference);
		}

		void visit_array_index(AST::array_index& value, ref) override {
			visit(value.lhs);
			visit(value.rhs);
		}

		void visit_call(AST::call& value, ref) override {
			visit(value.lhs);
			for(auto arg: value.elements)
				visit(arg);
		}

		void visit_float_literal(AST::float_literal&, ref) override {
			// Do nothing
		}

		void visit_int_literal(AST::int_literal&, ref) override {
			// Do nothing
		}

		void visit_string_literal(AST::string_literal&, ref) override {
			// Do nothing
		}

		void visit_bool_literal(AST::bool_literal&, ref) override {
			// Do nothing
		}

		void visit_none_literal(AST::none_literal&, ref) override {
			// Do nothing
		}

		void visit_list_literal(AST::list_literal& value, ref) override {
			for(auto elem: value.elements)
				visit(elem);
		}
	};

} // namespace AST