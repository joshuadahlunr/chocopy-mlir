#pragma once

#include "AST.hpp"
#include "peglib.h"

#include <cstddef>
#include <optional>
#include <cmath>
#include <span>
#include <string_view>

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
	size_t total_bytes = 0;

	while (std::getline(in, line)) {
		int col = 0;
		bool saw_space = false;
		bool saw_tab = false;
		size_t i;
		struct deferred_update {
			size_t& total_bytes;
			size_t& i;

			~deferred_update() {
				total_bytes += i;
			}
		} deferred = {total_bytes, i};

		// Measure indentation
		for (i = 0; i < line.size(); ++ i)
			if (line[i] == ' ') {
				col++;
				saw_space = true;
			} else if (line[i] == '\t') {
				col = next_tab_stop(col);
				saw_tab = true;
			} else break;

		if (saw_space && saw_tab) {
			diagnostics::singleton().push_error("Mixed tabs and spaces", line, {total_bytes + i, total_bytes + i + 1});
			return "";
		}

		std::string trimmed = line.substr(i);
		
		// Strip comments from the trimmed portion
		size_t comment_pos = trimmed.find('#');
		if (comment_pos != std::string::npos) {
			trimmed = trimmed.substr(0, comment_pos);
		}
		
		// Remove trailing whitespace after comment removal
		auto last_non_ws = trimmed.find_last_not_of(" \t\r\n\v\f\x85\xA0");
		if (last_non_ws != std::string::npos) {
			trimmed = trimmed.substr(0, last_non_ws + 1);
		} else {
			trimmed.clear();
		}

		if (!first) out << '\n';
		first = false;

		// Blank line (whitespace/comments only) → pass through unchanged
		if (trimmed.empty()) {
			out << line;
			continue;
		}

		// Apply trimming to end of line
		size_t end = line.size() - 1;
		if(last_non_ws != std::string::npos)
			end = i + last_non_ws + 1;
		else if(comment_pos != std::string::npos)
			end = i + comment_pos;
		line = line.substr(0, end);

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
			if (indent_stack.back() != col) {
				diagnostics::singleton().push_error("Unaligned dedent level", line, {total_bytes + i, total_bytes + i + 1});
				return "";
			}
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

	constexpr static auto make_location = [](const peg::SemanticValues &vs) {
		diagnostics::source_location out;
		out.start_byte = size_t(vs.sv().data());
		out.end_byte = size_t(vs.sv().data() + vs.sv().size());
		out.filename = vs.path ? std::string_view{vs.path} : std::string_view{};
		return out;
	};

	parser["program"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out = {{make_location(vs)}};
		for(auto& stmt: vs)
			out.elements.push_back(AST::a2r(stmt));
		return AST::make_node(ast, out);
	};

	parser["class_def"] = [&ast](const peg::SemanticValues &vs) {
		AST::class_declaration_lookup decl = {std::any_cast<AST::block>(vs[4])};
		decl.location = make_location(vs);
		decl.name = std::any_cast<interned_string>(vs[0]);
		decl.base = std::any_cast<interned_string>(vs[1]);
		return AST::make_node(ast, decl);
	};

	parser["class_body"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out = {{make_location(vs), AST::absent}};
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

		func.location = make_location(vs);
		func.name = name;
		func.return_type = return_type ? *return_type : AST::absent;
		if(params) {
			func.num_parameters = params->size();
			for(size_t i = func.num_parameters; i--; )
				func.elements.insert(func.elements.begin(), AST::make_node(ast, AST::parameter_declaration{
					{make_location(vs), AST::absent},
					(*params)[i].first, (*params)[i].second, i
				}));
		}

		return AST::make_node(ast, func);
	};

	parser["func_body"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out = {{make_location(vs), AST::absent}};
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
			make_location(vs), AST::absent, name, type, initial_value
		});
	};

	parser["global_decl"] = [&ast](const peg::SemanticValues &vs) {
		return AST::make_node(ast, AST::global_lookup{
			make_location(vs), AST::absent, std::any_cast<interned_string>(vs[0])
		});
	};

	parser["nonlocal_decl"] = [&ast](const peg::SemanticValues &vs) {
		return AST::make_node(ast, AST::nonlocal_lookup{
			make_location(vs), AST::absent, std::any_cast<interned_string>(vs[0])
		});
	};

	parser["typed_var"] = [&ast](const peg::SemanticValues &vs) {
		return std::make_pair(std::any_cast<interned_string>(vs[0]), AST::a2r(vs[1]));
	};

	parser["type"] = [&ast, identifier_parser = std::move(identifier_parser)](const peg::SemanticValues &vs) {
		switch(vs.choice()) {
		case 0: // IDENT
			return AST::make_node(ast, AST::type_lookup{make_location(vs), AST::absent, std::any_cast<interned_string>(vs[0])});
		case 1: { // STRING # NOTE: Required to be a valid IDENT!
			auto raw = std::any_cast<interned_string>(vs[0]);
			if(!identifier_parser.parse(raw))
				return AST::make_error(ast, diagnostics::singleton().push_error("Type strings should be proper identifiers!", vs.ss, make_location(vs)));
			return AST::make_node(ast, AST::type_lookup{make_location(vs), AST::absent, raw});
		}
		case 2: // '[' type ']'
			return AST::make_node(ast, AST::list_type{make_location(vs), AST::absent, AST::a2r(vs[0])});
		default: throw std::runtime_error("type unreachable");
		}
	};

	parser["simple_stmt"] = [&ast](const peg::SemanticValues &vs) {
		switch (vs.choice()) {
		case 0: // 'pass'
			return AST::make_node(ast, AST::pass_statement{make_location(vs), AST::absent});
		case 1: // 'return' expr?
			return AST::make_node(ast, AST::return_statement{make_location(vs), AST::absent, vs.size() ? AST::a2r(vs[0]) : AST::absent});
		case 2:{ // (target '=')* expr
			auto lhs = AST::a2r(vs[0]);
			for(size_t i = 1; i < vs.size(); ++i) {
				AST::assignment expr = {make_location(vs), AST::absent};
				expr.lhs = lhs; expr.rhs = AST::a2r(vs[i]);
				lhs = AST::make_node(ast, expr);
			}
			return lhs;
		}
		default: throw std::runtime_error("simple_stmt unreachable");
		}
	};

	parser["target"] = [&ast](const peg::SemanticValues &vs) {
		auto lookup = std::any_cast<interned_string>(vs[0]);
		auto lhs = AST::make_node(ast, AST::variable_store_lookup{{make_location(vs), AST::absent}, lookup});
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
		AST::if_statement stmt = {make_location(vs)};
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
		AST::for_statement_lookup stmt = {make_location(vs), AST::absent, std::any_cast<AST::block>(vs[2])};
		stmt.source = AST::a2r(vs[1]);
		stmt.interned_name = std::any_cast<interned_string>(vs[0]);
		return AST::make_node(ast, stmt);
	};

	parser["while_stmt"] = [&ast](const peg::SemanticValues &vs) {
		AST::while_statement stmt = {make_location(vs), AST::absent, std::any_cast<AST::block>(vs[1])};
		stmt.condition = AST::a2r(vs[0]);
		auto ref = AST::make_node(ast, stmt);
		return ref;
	};

	parser["block"] = [&ast](const peg::SemanticValues &vs) {
		AST::block out = {make_location(vs)};
		std::span relevant = vs;
		relevant = relevant.subspan(2, relevant.size() - 3); // Trim away the NEWLINE, INDENT, AND DEDENT
		for(auto& stmt: relevant)
			out.elements.push_back(AST::a2r(stmt));
		return out;
	};

	parser["if_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.size() == 1) // No if
			return vs[0];

		AST::if_expression expr = {make_location(vs)};
		expr.then = AST::a2r(vs[0]);
		expr.condition = AST::a2r(vs[1]);
		expr.else_ = AST::a2r(vs[2]);
		return AST::make_node(ast, expr);
	};

	parser["or_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			AST::logical_or expr = {make_location(vs)};
			expr.lhs = lhs; expr.rhs = AST::a2r(vs[i]);
			lhs = AST::make_node(ast, expr);
		}
		return lhs;
	};

	parser["and_expr"] = [&ast](const peg::SemanticValues &vs) {
		auto lhs = AST::a2r(vs[0]);
		for(size_t i = 1; i < vs.size(); ++i) {
			AST::logical_and expr = {make_location(vs)};
			expr.lhs = lhs; expr.rhs = AST::a2r(vs[i]);
			lhs = AST::make_node(ast, expr);
		}
		return lhs;
	};

	parser["not_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.choice() == 0) { // 'not' not_expr
			AST::logical_not not_ = {make_location(vs)};
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
			break; case '=': {
				AST::equal expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
			break; case '!': {
				AST::not_equal expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
			break; case '<':
				if(token.size() == 1){
					AST::less expr = {make_location(vs)};
					expr.lhs = lhs; expr.rhs = rhs;
					lhs = AST::make_node(ast, expr);
				} else {
					AST::less_equal expr = {make_location(vs)};
					expr.lhs = lhs; expr.rhs = rhs;
					lhs = AST::make_node(ast, expr);
				}
			break; case '>':
				if(token.size() == 1){
					AST::greater expr = {make_location(vs)};
					expr.lhs = lhs; expr.rhs = rhs;
					lhs = AST::make_node(ast, expr);
				} else {
					AST::greater_equal expr = {make_location(vs)};
					expr.lhs = lhs; expr.rhs = rhs;
					lhs = AST::make_node(ast, expr);
				}
			break; case 'i':{
				AST::is expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
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
			break; case '+': {
				AST::add expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
			break; case '-': {
				AST::subtract expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
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
			break; case '*': {
				AST::multiply expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
			break; case '/':
				if(token.size() == 1) {
					AST::divide expr = {make_location(vs)};
					expr.lhs = lhs; expr.rhs = rhs;
					lhs = AST::make_node(ast, expr);
				} else {
					AST::quotient expr = {make_location(vs)};
					expr.lhs = lhs; expr.rhs = rhs;
					lhs = AST::make_node(ast, expr);
				}
			break; case '%': {
				AST::remainder expr = {make_location(vs)};
				expr.lhs = lhs; expr.rhs = rhs;
				lhs = AST::make_node(ast, expr);
			}
			break; default: throw std::runtime_error("mul_expr unreachable");
			}
		}
		return lhs;
	};

	parser["unary_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.choice() == 0) { // '-' unary_expr
			AST::negate neg = {make_location(vs)};
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
		AST::member_access_lookup out = {make_location(vs)};
		out.interned_name = std::any_cast<interned_string>(vs[0]);
		return out;
	};

	parser["index_expr"] = [&ast](const peg::SemanticValues &vs) {
		AST::array_index out = {make_location(vs)};
		out.rhs = AST::a2r(vs[0]);
		return out;
	};

	parser["call_expr"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		if(vs.size())
			return vs[0];
		else return AST::call{make_location(vs)}; // argumentless call!
	};

	parser["arg_list"] = [&ast](const peg::SemanticValues &vs) {
		AST::call call = {make_location(vs)};
		for(auto& elem: vs)
			call.elements.push_back(AST::a2r(elem));
		return call;
	};

	parser["primary"] = [&ast](const peg::SemanticValues &vs) -> std::any {
		switch (vs.choice()) {
		case 1: { // IDENT
			AST::variable_load_lookup lookup = {make_location(vs)};
			lookup.interned_name = std::any_cast<interned_string>(vs[0]);
			return AST::make_node(ast, lookup);
		}
		default:
			return vs[0];
		}
	};

	parser["list_literal"] = [&ast](const peg::SemanticValues &vs) {
		AST::list_literal out = {make_location(vs)};
		for(auto& elem: vs)
			out.elements.push_back(AST::a2r(elem));
		return AST::make_node(ast, out);
	};

	parser["literal"] = [&ast, &interner](const peg::SemanticValues &vs) {
		AST::type_lookup type = {make_location(vs)};
		switch (vs.choice()) {
		case 0: // None
			type.interned_name = interner.intern("<NONE>");
			return AST::make_node(ast, AST::none_literal{make_location(vs), AST::absent, AST::make_node(ast, type)});
		case 1: // True
			type.interned_name = interner.intern("bool");
			return AST::make_node(ast, AST::bool_literal{make_location(vs), AST::absent, AST::make_node(ast, type), true});
		case 2: // False
			type.interned_name = interner.intern("bool");
			return AST::make_node(ast, AST::bool_literal{make_location(vs), AST::absent, AST::make_node(ast, type), false});
		case 5: // String
			type.interned_name = interner.intern("str");
			return AST::make_node(ast, AST::string_literal{make_location(vs), AST::absent, AST::make_node(ast, type), std::any_cast<interned_string>(vs[0])});
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

	parser["FLOAT"] = [&ast, &interner](const peg::SemanticValues &vs) {
		AST::type_lookup type = {make_location(vs)};
		AST::float_literal lit = {make_location(vs)};
		type.interned_name = interner.intern("float");
		lit.value = vs.token_to_number<double>();
		lit.type = AST::make_node(ast, type);
		return AST::make_node(ast, lit);
	};
	parser["INTEGER"] = [&ast, &interner](const peg::SemanticValues &vs) {
		AST::type_lookup type = {make_location(vs)};
		AST::int_literal lit = {make_location(vs)};
		type.interned_name = interner.intern("int");
		lit.value = vs.token_to_number<int64_t>();
		lit.type = AST::make_node(ast, type);
		return AST::make_node(ast, lit);
	};

    return std::move(parser);
}