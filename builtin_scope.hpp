#pragma once
#include "AST.hpp"
#include <string_view>

struct builtin_scope {
	AST::ref root;
	size_t size;

	// Types
	AST::ref __i64__, none, empty, object, Float, Int, Bool, Str;

	// Functions
	AST::ref print, len, free;
};

inline std::tuple<AST::flattened, string_interner, builtin_scope> initialize_builtin_block() {
	AST::flattened ast; ast.reserve(64);
	string_interner interner;
	AST::node_base builtin {{0, 0, interner.intern("<builtin>")}, AST::absent};
	builtin_scope scope;
	AST::block block = {builtin};
	scope.root = builtin.scope_block = AST::make_node(ast, std::move(block));
	AST::ref zero = AST::make_node(ast, AST::int_literal{builtin, AST::absent, 0});

	auto make_type = [&](std::string_view name, AST::ref base = AST::absent, size_t size = 64, size_t alignment = 64) -> AST::ref {
		AST::class_declaration type = {builtin};
		type.name = interner.intern(name);
		type.base = base;
		type.size = size;
		type.alignment = alignment;
		
		AST::ref out;
		block.elements.push_back(out = AST::make_node(ast, std::move(type)));
		return out;
	};
	auto make_pass_type = [&](std::string_view name, AST::ref base = AST::absent, size_t size = 64, size_t alignment = 64) -> AST::ref {
		auto out = make_type(name, base, size, alignment);
		ast.back().as_class_declaration().elements.push_back(AST::make_node(ast, AST::pass_statement{builtin}));
		return out;
	};

	auto make_function = [&](std::string_view name, AST::ref scope, size_t num_params, AST::ref return_type) {
		AST::function_declaration func = {builtin};
		func.scope_block = scope;
		func.name = interner.intern(name);
		func.num_parameters = num_params;
		func.return_type = return_type;
		func.elements = {
			AST::make_node(ast, AST::pass_statement{builtin})
		};
		return AST::make_node(ast, std::move(func));
	};

	// Type declarations
	scope.__i64__ = make_pass_type("<4BYTES>");

	scope.none = make_pass_type("<NONE>");
	AST::ref none_lit = AST::make_node(ast, AST::none_literal{builtin, scope.none});

	scope.empty = make_pass_type("<EMPTY>");

	scope.object = make_type("object", AST::absent, 64 * 2);
	scope.Float = make_type("float", scope.object, 64 * 2);
	scope.Int = make_type("int", scope.Float, 64 * 2);
	auto& object = ast[scope.object].as_class_declaration();
	object.elements.push_back(AST::make_node(ast, AST::variable_declaration{
		builtin, interner.intern("__tag__"), scope.__i64__, none_lit
	}));
	object.elements.push_back(make_function("__print__", scope.object, 1, scope.Int));
	object.elements.push_back(make_function("__len__", scope.object, 1, scope.Int));
	
	auto& Float = ast[scope.Float].as_class_declaration();
	Float.elements.push_back(AST::make_node(ast, AST::variable_declaration{
		builtin, interner.intern("__value__"), scope.__i64__, zero
	}));
	Float.elements.push_back(make_function("__print__", scope.Float, 1, scope.Int));

	ast[zero].as_int_literal().type = scope.Int;
	auto& Int = ast[scope.Int].as_class_declaration();
	Int.elements.push_back(make_function("__print__", scope.Int, 1, scope.Int));

	scope.Bool = make_type("bool", scope.object, 64 * 2);
	auto& Bool = ast[scope.Bool].as_class_declaration();
	Bool.elements.push_back(AST::make_node(ast, AST::variable_declaration{
		builtin, interner.intern("__bool__"), scope.__i64__, zero
	}));
	Bool.elements.push_back(make_function("__print__", scope.Int, 1, scope.Int));

	scope.Str = make_type("str", scope.object, 64 * 2);
	auto& Str = ast[scope.Str].as_class_declaration();
	Str.elements.push_back(AST::make_node(ast, AST::variable_declaration{
		builtin, interner.intern("__size__"), scope.__i64__, zero
	}));
	Str.elements.push_back(make_function("__print__", scope.Str, 1, scope.Int));
	Str.elements.push_back(make_function("__len__", scope.Str, 1, scope.Int));


	// Function Declarations
	block.elements.push_back(scope.print = make_function("print", scope.root, 1, scope.Int));
	block.elements.push_back(scope.len = make_function("len", scope.root, 1, scope.Int));
	block.elements.push_back(scope.free = make_function("free", scope.root, 1, scope.Int));

	ast[scope.root].as_block() = std::move(block); // Make sure all the updates to block get applied
	scope.size = ast.size();
	return {std::move(ast), std::move(interner), scope};
}