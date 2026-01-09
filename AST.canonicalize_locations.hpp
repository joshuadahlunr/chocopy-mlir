#include "AST.hpp"
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace AST {

	struct canonicalize_locations : public visiter<void> {
		std::string_view source;

		explicit canonicalize_locations(AST::flattened &ast, std::string_view source) : visiter<void>(ast), source(source) {}

		

		void visit_ref(ref&, ref) override {}

		void visit_type_lookup(type_lookup& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_list_type(list_type& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.type);
		}

		void visit_class_declaration_lookup(class_declaration_lookup& value, ref r) override {
			value.location.pointers_to_bytes(source);
			visit_block(value, r);
		}

		void visit_class_declaration(class_declaration& value, ref r) override {
			value.location.pointers_to_bytes(source);
			visit_block(value, r);
		}

		void visit_function_declaration(function_declaration& value, ref r) override {
			value.location.pointers_to_bytes(source);
			visit_block(value, r);
		}

		void visit_parameter_declaration(parameter_declaration& value, ref) override {
			value.location.pointers_to_bytes(source);
			if(value.type != absent) visit(value.type);
		}

		void visit_variable_declaration(variable_declaration& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.type);
			visit(value.initial_value);
		}

		void visit_global_lookup(global_lookup& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_global(global& value, ref) override {
			value.location.pointers_to_bytes(source);
			// visit(value.reference); // Causes cycle!
		}

		void visit_nonlocal_lookup(nonlocal_lookup& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_nonlocal(nonlocal& value, ref) override {
			value.location.pointers_to_bytes(source);
			// visit(value.reference); // Causes cycle!
		}

		void visit_pass_statement(pass_statement& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_return_statement(return_statement& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.what);
		}

		void visit_assignment(assignment& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
			visit(value.rhs);
		}

		void visit_if_statement(if_statement& value, ref r) override {
			value.location.pointers_to_bytes(source);
			for(auto& [condition, block]: value.condition_block_pairs) {
				if(condition != absent) visit(condition);
				visit_block(block, r);
			}
		}

		void visit_while_statement(while_statement& value, ref r) override {
			value.location.pointers_to_bytes(source);
			visit(value.condition);
			visit_block(value, r);
		}

		void visit_for_statement(for_statement& value, ref r) override {
			value.location.pointers_to_bytes(source);
			visit(value.source);
			visit_block(value, r);
		}

		void visit_block(block& value, ref r) override {
			value.location.pointers_to_bytes(source);
			for (auto &elem : value.elements) {
				visit(elem);
				ast[elem].as_node_base().scope_block = r;
			}
		}

		void visit_if_expression(if_expression& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.then);
			visit(value.condition);
			visit(value.else_);
		}

		void visit_logical_and(logical_and& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
			visit(value.rhs);
		}

		void visit_logical_or(logical_or& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_equal(equal& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_not_equal(not_equal& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_less(less& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_less_equal(less_equal& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_greater(greater& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_greater_equal(greater_equal& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_is(is& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_add(add& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_subtract(subtract& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_multiply(multiply& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_quotient(quotient& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_remainder(remainder& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_divide(divide& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_logical_not(logical_not& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.what); 
		}

		void visit_negate(negate& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.what); 
		}

		void visit_variable_load_lookup(variable_load_lookup& value, ref) override {
			value.location.pointers_to_bytes(source);
		}	

		void visit_variable_load(variable_load& value, ref) override {
			value.location.pointers_to_bytes(source);
			// visit(value.reference); // creates cycle
		}	

		void visit_variable_store_lookup(variable_store_lookup& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_variable_store(variable_store& value, ref) override {
			value.location.pointers_to_bytes(source);
			// visit(value.reference); // creates cycle
		}

		void visit_member_access_lookup(member_access_lookup& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs);
		}

		void visit_member_access(member_access& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs);
			// visit(value.reference); // creates cycle
		}

		void visit_array_index(array_index& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs); 
            visit(value.rhs);
		}

		void visit_call(call& value, ref) override {
			value.location.pointers_to_bytes(source);
			visit(value.lhs);
			for(auto arg: value.elements)
				visit(arg);
		}

		void visit_float_literal(float_literal& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_int_literal(int_literal& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_string_literal(string_literal& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_bool_literal(bool_literal& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_none_literal(none_literal& value, ref) override {
			value.location.pointers_to_bytes(source);
		}

		void visit_list_literal(list_literal& value, ref) override {
			value.location.pointers_to_bytes(source);
			for(auto elem: value.elements)
				visit(elem);
		}
	};

} // namespace AST