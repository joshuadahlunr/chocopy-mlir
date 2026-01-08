#pragma once

#include "AST.hpp"
#include "peglib.h"

#include <optional>
#include <cmath>
#include <span>

struct IndentError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

static int next_tab_stop(int col) {
	return col + (8 - (col % 8));
}

static std::string mark(std::string_view line, char bracket) {
	if (line.size() < 1) return std::string(1, bracket);
	auto unsafe = const_cast<char*>(line.data());
	unsafe[0] = bracket;
	return "";
}

inline std::string preprocess_indentation(const std::string& src) {
	std::istringstream in(src);
	std::ostringstream out;

	std::vector<int> indent_stack;
	indent_stack.push_back(0);

	std::string line;
	bool first = true;

	while (std::getline(in, line)) {
		int col = 0;
		bool saw_space = false;
		bool saw_tab = false;
		size_t i;

		// Measure indentation
		for (i = 0; i < line.size(); ++ i)
			if (line[i] == ' ') {
				col++;
				saw_space = true;
			} else if (line[i] == '\t') {
				col = next_tab_stop(col);
				saw_tab = true;
			} else break;

		if (saw_space && saw_tab)
			throw IndentError("Mixed tabs and spaces in indentation");

		std::string trimmed = line.substr(i);

		if (!first) out << '\n';
		first = false;

		// Blank line → pass through unchanged
		if (trimmed.empty()) {
			out << line;
			continue;
		}

		int prev = indent_stack.back();

		if (col > prev) {
			indent_stack.push_back(col);
			out << mark(line, '{');
			out << line;
		} else if (col < prev) {
			std::string dedent_line = line;
			std::string_view view = dedent_line;
			size_t first_non_whitespace = std::distance(dedent_line.begin(), std::find_if(dedent_line.begin(), dedent_line.end(), [](char c) {
				return !(c == ' ' || c == '\t');
			}));
			size_t replacements = 0;

			while (indent_stack.back() > col) {
				indent_stack.pop_back();
				if(replacements < first_non_whitespace)
					out << mark(view.substr(replacements, first_non_whitespace - replacements), '}');
				else out << '}';
				++replacements;
			}
			if (indent_stack.back() != col)
				throw IndentError("Unaligned dedent level");
			out << dedent_line;
		} else out << line;

		// Emit newlines
		auto last = line.find_last_not_of(" \t\r\n\v\f\x85\xA0");
		if(last != std::string::npos && line[last] != '\\')
			out << ';';
	}

	// Emit final DEDENTs
	while (indent_stack.size() > 1) {
		out << '}';
		indent_stack.pop_back();
	}

	return out.str();
}

inline std::tuple<AST::flattened, string_interner, AST::ref> initialize_builtin_block() {
	AST::flattened ast;
	string_interner interner;
	AST::block block;
	auto builtin_block = AST::make_node(ast, std::move(block));
	AST::ref __4bytes__, object_ref, global_none;
	{
		AST::class_declaration i32;
		i32.name = interner.intern("__4bytes__");
		i32.base = AST::absent;
		i32.size = 32;
		i32.alignment = 32;
		i32.elements.push_back(AST::make_node(ast, AST::pass_statement{}));
		block.elements.push_back(__4bytes__ = AST::make_node(ast, i32));
	}
	{
		AST::class_declaration object;
		object.name = interner.intern("object");
		object.base = AST::absent;
		object.size = 32 * 3;
		object.alignment = 32;
		object.elements = {
			AST::make_node(ast, AST::variable_declaration{
				interner.intern("__tag__"),
				__4bytes__, global_none = AST::make_node(ast, AST::none{})
			}), AST::make_node(ast, AST::variable_declaration{
				interner.intern("__size__"),
				__4bytes__, global_none
			}), AST::make_node(ast, AST::variable_declaration{
				interner.intern("__vtable__"),
				__4bytes__, global_none
			})
		};
		block.elements.push_back(object_ref = AST::make_node(ast, object));
	}
	{
		AST::class_declaration Int;
		Int.name = interner.intern("int");
		Int.base = object_ref;
		Int.size = 32 * (3 + 1);
		Int.alignment = 32;
		Int.elements = {
			AST::make_node(ast, AST::variable_declaration{
				interner.intern("__int__"),
				__4bytes__, AST::make_node(ast, double(0))
			})
		};
		block.elements.push_back(AST::make_node(ast, Int));
	}
	{
		AST::class_declaration Int;
		Int.name = interner.intern("float");
		Int.base = object_ref;
		Int.size = 32 * (3 + 1);
		Int.alignment = 32;
		Int.elements = {
			AST::make_node(ast, AST::variable_declaration{
				interner.intern("__float__"),
				__4bytes__, AST::make_node(ast, double(0))
			})
		};
		block.elements.push_back(AST::make_node(ast, Int));
	}
	{
		AST::class_declaration Bool;
		Bool.name = interner.intern("bool");
		Bool.base = object_ref;
		Bool.size = 32 * (3 + 1);
		Bool.alignment = 32;
		Bool.elements = {
			AST::make_node(ast, AST::variable_declaration{
				interner.intern("__bool__"),
				__4bytes__, AST::make_node(ast, double(0))
			})
		};
		block.elements.push_back(AST::make_node(ast, Bool));
	}
	{
		AST::class_declaration str;
		str.name = interner.intern("str");
		str.base = object_ref;
		str.size = 32 * (3 + 1);
		str.alignment = 32;
		str.elements = {
			AST::make_node(ast, AST::variable_declaration{
				interner.intern("__len__"),
				__4bytes__, AST::make_node(ast, double(0))
			})
		};
		block.elements.push_back(AST::make_node(ast, str));
	}
	{
		AST::class_declaration list;
		list.name = interner.intern("list");
		list.base = object_ref;
		list.size = 32 * (3 + 1);
		list.alignment = 32;
		list.elements = {
			AST::make_node(ast, AST::variable_declaration{
				interner.intern("__len__"),
				__4bytes__, AST::make_node(ast, double(0))
			})
		};
		block.elements.push_back(AST::make_node(ast, list));
	}
	ast[builtin_block].as_block() = std::move(block);
	return {std::move(ast), std::move(interner), builtin_block};
}

inline peg::parser initialize_parser(AST::flattened& ast, string_interner& interner){
    auto grammar =
#include "chocopy.peg"
	;

	peg::parser parser;
	parser.set_logger([](size_t line, size_t col, const std::string& msg, const std::string &rule) {
		std::cerr << line << ":" << col << ": " << msg << std::endl;
	});
	auto ok = parser.load_grammar(grammar);
	assert(ok);

	peg::parser identifier_parser(grammar, "IDENT");

	parser.enable_packrat_parsing(); // Enable packrat parsing.
	// parser.enable_trace([](const peg::Ope &name, const char *s, size_t n, const peg::SemanticValues &vs,
	// const peg::Context &c, const std::any &dt, std::any &trace_data){
	//     std::cout << "> " << vs.name() << std::endl;
	// }, [](const peg::Ope &ope, const char *s, size_t n, const peg::SemanticValues &vs,
	// const peg::Context &c, const std::any &dt, size_t, std::any &trace_data) {
	//     std::cout << "< " << vs.name() << std::endl;
	// });

	parser["program"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out;
		for(auto& stmt: vs)
			out.elements.push_back(AST::a2r(stmt));
		return AST::make_node(ast, out);
	};

	parser["class_def"] = [&ast](const peg::SemanticValues &vs) {
		AST::class_declaration_lookup decl = {std::any_cast<AST::block>(vs[4])};
		decl.name = std::any_cast<interned_string>(vs[0]);
		decl.base = std::any_cast<interned_string>(vs[1]);
		return AST::make_node(ast, decl);
	};

	parser["class_body"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out;
		if(vs.choice() == 0) { // 'pass' NEWLINE
			out.elements.push_back(AST::make_node(ast, AST::pass_statement{}));
			return out;
		}

		for(auto& decl: vs)
			out.elements.push_back(AST::a2r(decl));
		return out;
	};

	parser["func_def"] = [&ast](const peg::SemanticValues &vs) {
		interned_string name = std::any_cast<interned_string>(vs[0]);
		std::optional<std::vector<std::pair<interned_string, AST::ref>>> params = {};
		std::optional<AST::ref> return_type = {};
		AST::function_declaration func;
		switch (vs.size()) {
		break; case 5:
			func = {std::any_cast<AST::block>(vs[3])};
		break; case 6:
			if(vs[1].type() == typeid(AST::ref))
				return_type = AST::a2r(vs[1]);
			else params = std::any_cast<std::vector<std::pair<interned_string, AST::ref>>>(vs[1]);
			func = {std::any_cast<AST::block>(vs[4])};
		break; case 7:
			params = std::any_cast<std::vector<std::pair<interned_string, AST::ref>>>(vs[1]);
			return_type = AST::a2r(vs[2]);
			func = {std::any_cast<AST::block>(vs[5])};
		break; default: throw std::runtime_error("func_def unreachable");
		}

		func.name = name;
		func.return_type = return_type ? *return_type : AST::absent;
		if(params) {
			func.num_parameters = params->size();
			for(size_t i = func.num_parameters; i--; )
				func.elements.insert(func.elements.begin(), AST::make_node(ast, AST::parameter_declaration{
					.name = (*params)[i].first,
					.type = (*params)[i].second,
					.index = i
				}));
		}
		
		return AST::make_node(ast, func);
	};

	parser["func_body"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out;
		for(auto& stmt: vs)
			out.elements.push_back(AST::a2r(stmt));
		return out;
	};

	parser["param_list"] = [&ast](const peg::SemanticValues &vs) {
		std::vector<std::pair<interned_string, AST::ref>> out;
		for(auto& typed_var: vs)
			out.emplace_back(std::any_cast<std::pair<interned_string, AST::ref>>(typed_var));
		return out;
	};

	parser["var_def"] = [&ast](const peg::SemanticValues &vs) {
		auto [name, type] = std::any_cast<std::pair<interned_string, AST::ref>>(vs[0]);
		auto initial_value = AST::a2r(vs[1]);
		return AST::make_node(ast, AST::variable_declaration{
			name, type, initial_value
		});
	};

	parser["global_decl"] = [&ast](const peg::SemanticValues &vs) {
		return AST::make_node(ast, AST::global_lookup{
			std::any_cast<interned_string>(vs[0])
		});
	};

	parser["nonlocal_decl"] = [&ast](const peg::SemanticValues &vs) {
		return AST::make_node(ast, AST::nonlocal_lookup{
			std::any_cast<interned_string>(vs[0])
		});
	};

	parser["typed_var"] = [&ast](const peg::SemanticValues &vs) {
		return std::make_pair(std::any_cast<interned_string>(vs[0]), AST::a2r(vs[1]));
	};

	parser["type"] = [&ast, identifier_parser = std::move(identifier_parser)](const peg::SemanticValues &vs) {
		switch(vs.choice()) {
		case 0: // IDENT
			return AST::make_node(ast, AST::type_lookup{std::any_cast<interned_string>(vs[0])});
		case 1: { // STRING # NOTE: Required to be a valid IDENT!
			auto raw = std::any_cast<interned_string>(vs[0]);
			if(!identifier_parser.parse(raw))
				throw std::runtime_error("Type strings should be proper identifiers!");
			return AST::make_node(ast, AST::type_lookup{raw});
		}
		case 2: // '[' type ']'
			return AST::make_node(ast, AST::list_type{AST::a2r(vs[0])});
		default: throw std::runtime_error("type unreachable");
		}
	};

	parser["simple_stmt"] = [&ast](const peg::SemanticValues &vs) {
		switch (vs.choice()) {
		case 0: // 'pass'
			return AST::make_node(ast, AST::pass_statement{});
		case 1: // 'return' expr?
			return AST::make_node(ast, AST::return_statement{vs.size() ? AST::a2r(vs[0]) : AST::absent});
		case 2:{ // (target '=')* expr
			auto lhs = AST::a2r(vs[0]);
			for(size_t i = 1; i < vs.size(); ++i)
				lhs = AST::make_node(ast, AST::assignment{lhs, AST::a2r(vs[i])});
			return lhs;
		}
		default: throw std::runtime_error("simple_stmt unreachable");
		}
	};

	parser["target"] = [&ast](const peg::SemanticValues &vs) {
		auto lookup = std::any_cast<interned_string>(vs[0]);
		auto lhs = AST::make_node(ast, AST::variable_store_lookup{lookup});
		for(size_t i = 1; i < vs.size(); ++i)
			if(vs[i].type() == typeid(AST::array_index)) {
				auto index = std::any_cast<AST::array_index>(vs[i]);
				index.lhs = lhs;
				lhs = AST::make_node(ast, index);
			} else if(vs[i].type() == typeid(AST::member_access_lookup)) {
				auto access = std::any_cast<AST::member_access_lookup>(vs[i]);
				access.lhs = lhs;
				lhs = AST::make_node(ast, access);
			} else throw std::runtime_error("target unreachable");
		return lhs;
	};

	parser["if_stmt"] = [&ast](const peg::SemanticValues &vs) {
		AST::if_statement stmt;
		size_t elseless_size = std::floor(vs.size() / 2) * 2;
		for(size_t i = 0; i < elseless_size; i += 2) {
			stmt.condition_block_pairs.emplace_back(
				AST::a2r(vs[i]),
				std::any_cast<AST::block>(vs[i + 1])
			);
		}
		if(elseless_size != vs.size())
			stmt.condition_block_pairs.emplace_back(
				AST::absent,
				std::any_cast<AST::block>(vs.back())
			);
		return AST::make_node(ast, stmt);
	};

	parser["for_stmt"] = [&ast](const peg::SemanticValues &vs) {
		AST::for_statement stmt = {std::any_cast<AST::block>(vs[2])};
		stmt.iterator = std::any_cast<interned_string>(vs[0]);
		stmt.source = AST::a2r(vs[1]);
		return AST::make_node(ast, stmt);
	};

	parser["while_stmt"] = [&ast](const peg::SemanticValues &vs) {
		AST::while_statement stmt = {std::any_cast<AST::block>(vs[1])};
		stmt.condition = AST::a2r(vs[0]);
		return AST::make_node(ast, stmt);
	};

	parser["block"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out;
		std::span relevant = vs;
		relevant = relevant.subspan(2, relevant.size() - 3); // Trim away the NEWLINE, INDENT, AND DEDENT
		for(auto& stmt: relevant)
			out.elements.push_back(AST::a2r(stmt));
		return out;
	};

	parser["if_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.size() == 1) // No if
			return vs[0];

		AST::if_expression expr;
		expr.then = AST::a2r(vs[0]);
		expr.condition = AST::a2r(vs[1]);
		expr.else_ = AST::a2r(vs[2]);
		return AST::make_node(ast, expr);
	};

	parser["or_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			auto rhs = AST::a2r(vs[i]);
			lhs = AST::make_node(ast, AST::logical_or{lhs, rhs});
		}
		return lhs;
	};

	parser["and_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			auto rhs = AST::a2r(vs[i]);
			lhs = AST::make_node(ast, AST::logical_and{lhs, rhs});
		}
		return lhs;
	};

	parser["not_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.choice() == 0) { // 'not' not_expr
			AST::logical_not not_;
			not_.what = AST::a2r(vs[0]);
			return AST::make_node(ast, not_);
		} else return vs[0];
	};

	parser["compare_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			auto token = vs.token(i - 1);
			auto rhs = AST::a2r(vs[i]);
			switch(token[0]){
			break; case '=':
				lhs = AST::make_node(ast, AST::equal{lhs, rhs});
			break; case '!':
				lhs = AST::make_node(ast, AST::not_equal{lhs, rhs});
			break; case '<':
				if(token.size() == 1)
					lhs = AST::make_node(ast, AST::less{lhs, rhs});
				else lhs = AST::make_node(ast, AST::less_equal{lhs, rhs});
			break; case '>':
				if(token.size() == 1)
					lhs = AST::make_node(ast, AST::greater{lhs, rhs});
				else lhs = AST::make_node(ast, AST::greater_equal{lhs, rhs});
			break; case 'i':
				lhs = AST::make_node(ast, AST::is{lhs, rhs});
			break; default: throw std::runtime_error("compare_expr unreachable");
			}
		}
		return lhs;
	};

	parser["add_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			auto token = vs.token(i - 1);
			auto rhs = AST::a2r(vs[i]);
			switch(token[0]){
			break; case '+':
				lhs = AST::make_node(ast, AST::add{lhs, rhs});
			break; case '-':
				lhs = AST::make_node(ast, AST::subtract{lhs, rhs});
			break; default: throw std::runtime_error("add_expr unreachable");
			}
		}
		return lhs;
	};

	parser["mul_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			auto token = vs.token(i - 1);
			auto rhs = AST::a2r(vs[i]);
			switch(token[0]){
			break; case '*':
				lhs = AST::make_node(ast, AST::multiply{lhs, rhs});
			break; case '/':
				if(token.size() == 1)
					lhs = AST::make_node(ast, AST::divide{lhs, rhs});
				else lhs = AST::make_node(ast, AST::quotient{lhs, rhs});
			break; case '%':
				lhs = AST::make_node(ast, AST::remainder{lhs, rhs});
			break; default: throw std::runtime_error("mul_expr unreachable");
			}
		}
		return lhs;
	};

	parser["unary_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.choice() == 0) { // '-' unary_expr
			AST::negate neg;
			neg.what = AST::a2r(vs[0]);
			return AST::make_node(ast, neg);
		} else return vs[0];
	};

	parser["postfix_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i)
			if(vs[i].type() == typeid(AST::call)) {
				auto call = std::any_cast<AST::call>(vs[i]);
				call.lhs = lhs;
				lhs = AST::make_node(ast, call);
			} else if(vs[i].type() == typeid(AST::array_index)) {
				auto index = std::any_cast<AST::array_index>(vs[i]);
				index.lhs = lhs;
				lhs = AST::make_node(ast, index);
			} else if(vs[i].type() == typeid(AST::member_access_lookup)) {
				auto access = std::any_cast<AST::member_access_lookup>(vs[i]);
				access.lhs = lhs;
				lhs = AST::make_node(ast, access);
			} else throw std::runtime_error("postfix_expr unreachable");
		return lhs;
	};

	parser["member_expr"] = [&ast](const peg::SemanticValues &vs) {
		AST::member_access_lookup out;
		out.interned_name = std::any_cast<interned_string>(vs[0]);
		return out;
	};

	parser["index_expr"] = [&ast](const peg::SemanticValues &vs) {
		AST::array_index out;
		out.rhs = AST::a2r(vs[0]);
		return out;
	};

	parser["call_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.size())
			return vs[0];
		else return AST::call{}; // argumentless call!
	};

	parser["arg_list"] = [&ast](const peg::SemanticValues &vs) {
		AST::call call;
		for(auto& elem: vs)
			call.elements.push_back(AST::a2r(elem));
		return call;
	};

	parser["primary"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		switch (vs.choice()) {
		case 1: { // IDENT
			AST::variable_load_lookup lookup = {std::any_cast<interned_string>(vs[0])};
			return AST::make_node(ast, lookup);
		}
		default:
			return vs[0];
		}
	};

	parser["list_literal"] = [&ast](const peg::SemanticValues &vs) {
		AST::list_literal out;
		for(auto& elem: vs)
			out.elements.push_back(AST::a2r(elem));
		return AST::make_node(ast, out);
	};

	parser["literal"] = [&ast](const peg::SemanticValues &vs) {
		switch (vs.choice()) {
		case 0: // None
			return AST::make_node(ast, AST::none{});
		case 1: // True
			return AST::make_node(ast, true);
		case 2: // False
			return AST::make_node(ast, false);
		case 5: // String
			return AST::make_node(ast, std::any_cast<interned_string>(vs[0]));
		default:
			return AST::a2r(vs[0]);
		}
	};

	parser["STRING"] = [&ast, &interner](const peg::SemanticValues &vs) {
		auto raw = vs.token(0);
		return interner.intern(decode_python_string(raw.substr(1, raw.size() - 2)));
	};
	parser["IDENT"] = [&ast, &interner](const peg::SemanticValues &vs) {
		return interner.intern(vs.token(0));
	};

	parser["FLOAT"] = [&ast](const peg::SemanticValues &vs) {
		return AST::make_node(ast, vs.token_to_number<double>());
	};
	parser["INTEGER"] = [&ast](const peg::SemanticValues &vs) {
		return AST::make_node(ast, vs.token_to_number<double>());
	};

    return std::move(parser);
}