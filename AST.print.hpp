#pragma once

#include "AST.hpp"
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

namespace AST {

	struct pretty_printer : public visiter<std::string> {
		AST::flattened &ast;
		size_t indent = -2;
		bool name_only = false;

		explicit pretty_printer(AST::flattened &ast) : ast(ast) {}

		std::string gen_indent(size_t level) { 
			constexpr size_t min = -2;
			if(level > min) level = 0;
			return std::string(level, '\t'); 
		}

		template<typename Tfunction>
		auto name_only_block(const Tfunction& func) {
			auto back = name_only; name_only = true;
			auto out = func();
			name_only = back;
			return out;
		}

		std::string visit(ref ref) {
			return visiter<std::string>::visit(ast[ref], ref);
		}



		std::string visit_ref(const ref&, ref) override { return "<invalid>"; }

		std::string visit_type_lookup(const type_lookup& type, ref) override {
			return "lookup(" + std::string(type.interned_name) + ")";
		}

		std::string visit_list_type(const list_type& type, ref) override {
			return "[" + visit(type.type) + "]";
		}

		std::string visit_class_declaration_lookup(const class_declaration_lookup& decl, ref r) override {
			return "class " + std::string(decl.name) + "(lookup(" + std::string(decl.base) + ")):\n"
				+ visit_block(decl, r);
		}

		std::string visit_class_declaration(const class_declaration& decl, ref r) override {
			if(name_only) return std::string(decl.name);

			std::string base = "<ABSENT>";
			if(decl.base != AST::absent)
				base = ast[decl.base].as_class_declaration().name; 
			return "class " + std::string(decl.name) + "(" + base + "):\n"
				+ visit_block(decl, r);
		}

		std::string visit_function_declaration(const function_declaration& decl, ref r) override {
			std::string out = "def " + std::string(decl.name) + "(";
			if(decl.num_parameters) out += "params";
			out += "):\n" + visit_block(decl, r);
			return out;
		}

		std::string visit_parameter_declaration(const parameter_declaration& decl, ref) override {
			return name_only_block([&, this]{
				return "param(" + std::to_string(decl.index) + ", " + std::string(decl.name) + "): " + visit(decl.type);
			});
		}

		std::string visit_variable_declaration(const variable_declaration& decl, ref) override {
			return name_only_block([&, this]{
				return std::string(decl.name) + ": " + visit(decl.type) + " = " + visit(decl.initial_value);
			});
		}

		std::string visit_global_lookup(const global_lookup& decl, ref) override {
			return "global lookup(" + std::string(decl.interned_name) + ")";
		}

		std::string visit_global(const global& decl, ref) override {
			// return "global lookup(" + std::string(decl.interned_name) + ")";
			throw std::runtime_error("Not implemented yet!");
		}

		std::string visit_nonlocal_lookup(const nonlocal_lookup& decl, ref) override {
			return "nonlocal lookup(" + std::string(decl.interned_name) + ")";
		}

		std::string visit_nonlocal(const nonlocal& decl, ref) override {
			// return "nonlocal lookup(" + std::string(decl.interned_name) + ")";
			throw std::runtime_error("Not implemented yet!");
		}

		std::string visit_pass_statement(const pass_statement &, ref) override {
			return "pass";
		}

		std::string visit_return_statement(const return_statement& stmt, ref) override {
			return "return " + visit(stmt.what);
		}

		std::string visit_assignment(const assignment& expr, ref) override {
			return visit(expr.lhs) + " = " + visit(expr.rhs);
		}

		std::string visit_if_statement(const if_statement& stmt, ref r) override {
			auto& first = stmt.condition_block_pairs.front();
			std::span elifs = stmt.condition_block_pairs;
			bool else_present = false;
			if(elifs.back().condition == AST::absent) 
				elifs = elifs.subspan(1, elifs.size() - 2), else_present = true;
			else elifs = elifs.subspan(1);

			std::string out = "if " + visit(first.condition) + ":\n"
				+ visit_block(first.block, r);
			for(auto& elif: elifs)
				out += gen_indent(indent) + "if " + visit(elif.condition) + ":\n"
					+ visit_block(elif.block, r);
			if(else_present) 
				out += gen_indent(indent) + "else:\n"
					+ visit_block(stmt.condition_block_pairs.back().block, r);
			
			return out;
		}

		std::string visit_while_statement(const while_statement& stmt, ref r) override {
			return "while " + visit(stmt.condition) + ":\n"
				+ visit_block(stmt, r);
		}

		std::string visit_for_statement(const for_statement& stmt, ref r) override {
			return "for " + std::string(stmt.iterator) + " in " + visit(stmt.source) + ":\n"
				+ visit_block(stmt, r);
		}

		std::string visit_block(const block &block, ref) override {
			++indent;
			std::string out;
			for (auto &elem : block.elements)
			out += gen_indent(indent) + visit(elem) + "\n";
			--indent;
			return out;
		}

		std::string visit_if_expression(const if_expression& expr, ref) override {
			return "(" + visit(expr.then) + " if " + visit(expr.condition) + " else " + visit(expr.else_) + ")";
		}

		std::string visit_logical_and(const logical_and& expr, ref) override {
			return "(" + visit(expr.lhs) + " and " + visit(expr.rhs) + ")";
		}

		std::string visit_logical_or(const logical_or& expr, ref) override {
			return "(" + visit(expr.lhs) + " or " + visit(expr.rhs) + ")";
		}

		std::string visit_equal(const equal& expr, ref) override {
			return "(" + visit(expr.lhs) + " == " + visit(expr.rhs) + ")";
		}

		std::string visit_not_equal(const not_equal& expr, ref) override {
			return "(" + visit(expr.lhs) + " != " + visit(expr.rhs) + ")";
		}

		std::string visit_less(const less& expr, ref) override {
			return "(" + visit(expr.lhs) + " < " + visit(expr.rhs) + ")";
		}

		std::string visit_less_equal(const less_equal& expr, ref) override {
			return "(" + visit(expr.lhs) + " <= " + visit(expr.rhs) + ")";
		}

		std::string visit_greater(const greater& expr, ref) override {
			return "(" + visit(expr.lhs) + " > " + visit(expr.rhs) + ")";
		}

		std::string visit_greater_equal(const greater_equal& expr, ref) override {
			return "(" + visit(expr.lhs) + " >= " + visit(expr.rhs) + ")";
		}

		std::string visit_is(const is& expr, ref) override {
			return "(" + visit(expr.lhs) + " is " + visit(expr.rhs) + ")";
		}

		std::string visit_add(const add& expr, ref) override {
			return "(" + visit(expr.lhs) + " + " + visit(expr.rhs) + ")";
		}

		std::string visit_subtract(const subtract& expr, ref) override {
			return "(" + visit(expr.lhs) + " - " + visit(expr.rhs) + ")";
		}

		std::string visit_multiply(const multiply& expr, ref) override {
			return "(" + visit(expr.lhs) + " * " + visit(expr.rhs) + ")";
		}

		std::string visit_quotient(const quotient& expr, ref) override {
			return "(" + visit(expr.lhs) + " // " + visit(expr.rhs) + ")";
		}

		std::string visit_remainder(const remainder& expr, ref) override {
			return "(" + visit(expr.lhs) + " % " + visit(expr.rhs) + ")";
		}

		std::string visit_divide(const divide& expr, ref) override {
			return "(" + visit(expr.lhs) + " / " + visit(expr.rhs) + ")";
		}

		std::string visit_logical_not(const logical_not& expr, ref) override {
			return "(not" + visit(expr.what) + ")";
		}

		std::string visit_negate(const negate& expr, ref) override {
			return "(-" + visit(expr.what) + ")";
		}

		std::string visit_variable_load_lookup(const variable_load_lookup& expr, ref) override {
			return "lookup(" + std::string(expr.interned_name) + ")";
		}	

		std::string visit_variable_load(const variable_load& expr, ref) override {
			// return "lookup(" + std::string(expr.interned_name) + ")";
			throw std::runtime_error("Not implemented yet!");
		}	

		std::string visit_variable_store_lookup(const variable_store_lookup& expr, ref) override {
			return "lookup(" + std::string(expr.interned_name) + ")";
		}

		std::string visit_variable_store(const variable_store& expr, ref) override {
			// return "lookup(" + std::string(expr.interned_name) + ")";
			throw std::runtime_error("Not implemented yet!");
		}

		std::string visit_member_access_lookup(const member_access_lookup& expr, ref) override {
			return visit(expr.lhs) + ".lookup(" + std::string(expr.interned_name) + ")"; 
		}

		std::string visit_member_access(const member_access& expr, ref) override {
			// return visit(expr.lhs) + ".lookup(" + std::string(expr.interned_name) + ")"; 
			throw std::runtime_error("Not implemented yet!");
		}

		std::string visit_array_index(const array_index& expr, ref) override {
			return visit(expr.lhs) + "[" + visit(expr.rhs) + "]";
		}

		std::string visit_call(const call& expr, ref) override {
			std::string out = visit(expr.lhs) + "(";
			for(auto arg: expr.elements)
				out += visit(arg) + ", ";
			return out += ")";
		}

		std::string visit_double(const double& value, ref) override {
			std::ostringstream tmp;
			tmp << value;
			return tmp.str();
		}

		std::string visit_interned_string(const interned_string& value, ref) override {
			return '"' + std::string(value) + '"';
		}

		std::string visit_bool(const bool& value, ref) override {
			if(value) return "True";
			else return "False";
		}

		std::string visit_none(const none&, ref) override {
			return "None";
		}

		std::string visit_list_literal(const list_literal& lit, ref) override {
			std::string out = "[";
			for(auto elem: lit.elements)
				out += visit(elem) + ", ";
			return out += "]";
		}
	};

} // namespace AST