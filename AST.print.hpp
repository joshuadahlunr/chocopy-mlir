#pragma once

#include "AST.hpp"
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace AST {

	struct pretty_printer : public visiter<std::string> {
		std::unordered_set<ref> visited;
		size_t indent = -2;
		bool name_only = false;

		using visiter<std::string>::visiter;

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

		void check_cycle(ref r) {
			if(visited.contains(r))
				throw std::runtime_error("Tree cycle detected!");
			visited.insert(r);
		}



		std::string visit_ref(ref&, ref) override { return "<invalid>"; }

		std::string visit_type_lookup(type_lookup& type, ref r) override {
			check_cycle(r);
			return "lookup(" + std::string(type.interned_name) + ")";
		}

		std::string visit_list_type(list_type& type, ref r) override {
			if(!name_only) check_cycle(r);
			return "[" + visit(type.type) + "]";
		}

		std::string visit_function_type(function_type& type, ref r) override {
			if(!name_only) check_cycle(r);
			std::string out = "(";
			for(auto arg: type.elements)
				if(arg == AST::absent)
					out += "<ABSENT>, ";
				else out += visit(arg) + ", ";
			out += ")";
			if(type.return_type != AST::absent) out += " -> " + visit(type.return_type);
			return out;
		}

		std::string visit_class_declaration_lookup(class_declaration_lookup& decl, ref r) override {
			check_cycle(r);
			return "class " + std::string(decl.name) + "(lookup(" + std::string(decl.base) + ")):\n"
				+ visit_block(decl, r);
		}

		std::string visit_class_declaration(class_declaration& decl, ref r) override {
			if(name_only) return std::string(decl.name) + "->" + std::to_string(r);

			check_cycle(r);
			std::string base = "<ABSENT>";
			if(decl.base != AST::absent)
				base = name_only_block([&]() { return visit(decl.base); });
			return "class " + std::string(decl.name) + "(" + base + "):\n"
				+ visit_block(decl, r);
		}

		std::string visit_function_declaration(function_declaration& decl, ref r) override {
			if(name_only) return std::string(decl.name) + "->" + std::to_string(r);

			check_cycle(r);
			std::string out = "def " + std::string(decl.name) + "(";
			if(decl.num_parameters) out += "params n=" + std::to_string(decl.num_parameters);
			out += "):\n" + visit_block(decl, r);
			return out;
		}

		std::string visit_parameter_declaration(parameter_declaration& decl, ref r) override {
			if(name_only) return std::string(decl.name) + "->" + std::to_string(r);

			check_cycle(r);
			return name_only_block([&, this]{
				std::string out = "param(" + std::to_string(decl.index) + ", " + std::string(decl.name) + "): ";
				if(decl.type == absent) out += "<ABSENT>";
				else out += visit(decl.type);
				return out;
			});
		}

		std::string visit_variable_declaration(variable_declaration& decl, ref r) override {
			if(name_only) return std::string(decl.name) + "->" + std::to_string(r);

			check_cycle(r);
			return name_only_block([&, this]{
				return std::string(decl.name) + ": " + visit(decl.type) + " = " + visit(decl.initial_value);
			});
		}

		std::string visit_global_lookup(global_lookup& decl, ref r) override {
			if(name_only) return std::string(decl.interned_name) + "->" + std::to_string(r);

			check_cycle(r);
			return "global lookup(" + std::string(decl.interned_name) + ")";
		}

		std::string visit_global(global& decl, ref r) override {
			if(name_only) return visit(decl.reference) + "->" + std::to_string(r);

			check_cycle(r);
			return "global " + name_only_block([&]() {
				return visit(decl.reference);
			});
		}

		std::string visit_nonlocal_lookup(nonlocal_lookup& decl, ref r) override {
			if(name_only) return std::string(decl.interned_name) + "->" + std::to_string(r);

			check_cycle(r);
			return "nonlocal lookup(" + std::string(decl.interned_name) + ")";
		}

		std::string visit_nonlocal(nonlocal& decl, ref r) override {
			if(name_only) return visit(decl.reference) + "->" + std::to_string(r);

			check_cycle(r);
			return "nonlocal " + name_only_block([&]() {
				return visit(decl.reference);
			});
		}

		std::string visit_pass_statement(pass_statement &, ref r) override {
			check_cycle(r);
			return "pass";
		}

		std::string visit_return_statement(return_statement& stmt, ref r) override {
			check_cycle(r);
			return "return " + visit(stmt.what);
		}

		std::string visit_assignment(assignment& expr, ref r) override {
			check_cycle(r);
			return visit(expr.lhs) + " = " + visit(expr.rhs);
		}

		std::string visit_if_statement(if_statement& stmt, ref r) override {
			check_cycle(r);
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

		std::string visit_while_statement(while_statement& stmt, ref r) override {
			check_cycle(r);
			return "while " + visit(stmt.condition) + ":\n"
				+ visit_block(stmt, r);
		}

		std::string visit_for_statement(for_statement& stmt, ref r) override {
			check_cycle(r);
			return "for param(0) in " + visit(stmt.source) + ":\n"
				+ visit_block(stmt, r);
		}

		std::string visit_block(block &block, ref r) override {
			// check_cycle(r); // Function gets reused so should often trigger!
			++indent;
			std::string out;
			for (auto &elem : block.elements)
				out += gen_indent(indent) + visit(elem) + "\n";
			--indent;
			return out;
		}

		std::string visit_if_expression(if_expression& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.then) + " if " + visit(expr.condition) + " else " + visit(expr.else_) + ")";
		}

		std::string visit_explicit_cast(explicit_cast& value, ref r) override {
			check_cycle(r);
			return "(" + visit(value.reference) + " as " + name_only_block([&] {
				return visit(value.type);
			}) + ")";
		}

		std::string visit_logical_and(logical_and& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " and " + visit(expr.rhs) + ")";
		}

		std::string visit_logical_or(logical_or& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " or " + visit(expr.rhs) + ")";
		}

		std::string visit_equal(equal& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " == " + visit(expr.rhs) + ")";
		}

		std::string visit_not_equal(not_equal& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " != " + visit(expr.rhs) + ")";
		}

		std::string visit_less(less& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " < " + visit(expr.rhs) + ")";
		}

		std::string visit_less_equal(less_equal& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " <= " + visit(expr.rhs) + ")";
		}

		std::string visit_greater(greater& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " > " + visit(expr.rhs) + ")";
		}

		std::string visit_greater_equal(greater_equal& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " >= " + visit(expr.rhs) + ")";
		}

		std::string visit_is(is& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " is " + visit(expr.rhs) + ")";
		}

		std::string visit_add(add& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " + " + visit(expr.rhs) + ")";
		}

		std::string visit_subtract(subtract& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " - " + visit(expr.rhs) + ")";
		}

		std::string visit_multiply(multiply& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " * " + visit(expr.rhs) + ")";
		}

		std::string visit_quotient(quotient& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " // " + visit(expr.rhs) + ")";
		}

		std::string visit_remainder(remainder& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " % " + visit(expr.rhs) + ")";
		}

		std::string visit_divide(divide& expr, ref r) override {
			check_cycle(r);
			return "(" + visit(expr.lhs) + " / " + visit(expr.rhs) + ")";
		}

		std::string visit_logical_not(logical_not& expr, ref r) override {
			check_cycle(r);
			return "(not" + visit(expr.what) + ")";
		}

		std::string visit_negate(negate& expr, ref r) override {
			check_cycle(r);
			return "(-" + visit(expr.what) + ")";
		}

		std::string visit_variable_load_lookup(variable_load_lookup& expr, ref r) override {
			check_cycle(r);
			return "lookup(" + std::string(expr.interned_name) + ")";
		}

		std::string visit_variable_load(variable_load& expr, ref r) override {
			check_cycle(r);
			return name_only_block([&]() { return visit(expr.reference); });
		}

		std::string visit_variable_store_lookup(variable_store_lookup& expr, ref) override {
			return "lookup(" + std::string(expr.interned_name) + ")";
		}

		std::string visit_variable_store(variable_store& expr, ref r) override {
			check_cycle(r);
			return name_only_block([&]() { return visit(expr.reference); });
		}

		std::string visit_member_access_lookup(member_access_lookup& expr, ref r) override {
			check_cycle(r);
			return visit(expr.lhs) + ".lookup(" + std::string(expr.interned_name) + ")";
		}

		std::string visit_member_access(member_access& expr, ref r) override {
			check_cycle(r);
			return visit(expr.lhs) + "." + name_only_block([&]{
				return visit(expr.reference);
			});
		}

		std::string visit_array_index(array_index& expr, ref r) override {
			check_cycle(r);
			return visit(expr.lhs) + "[" + visit(expr.rhs) + "]";
		}

		std::string visit_call(call& expr, ref r) override {
			check_cycle(r);
			std::string out = visit(expr.lhs) + "(";
			for(auto arg: expr.elements)
				out += visit(arg) + ", ";
			return out += ")";
		}

		std::string visit_float_literal(float_literal& value, ref r) override {
			check_cycle(r);
			std::ostringstream out;
			out << value.value;
			return out.str();
		}

		std::string visit_int_literal(int_literal& value, ref r) override {
			return std::to_string(value.value);
		}

		std::string visit_string_literal(string_literal& value, ref r) override {
			return '"' + std::string(value.value) + '"';
		}

		std::string visit_bool_literal(bool_literal& value, ref r) override {
			return value.value ? "True" : "False";
		}

		std::string visit_none_literal(none_literal& value, ref r) override {
			return "None";
		}

		std::string visit_list_literal(list_literal& lit, ref r) override {
			check_cycle(r);
			std::string out = "[";
			for(auto elem: lit.elements)
				out += visit(elem) + ", ";
			return out += "]";
		}
	};

} // namespace AST