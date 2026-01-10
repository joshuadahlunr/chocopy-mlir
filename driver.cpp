
#include <iostream>

#include "AST.print.hpp" // Print pass
#include "AST.canonicalize_locations.hpp" // Update locaions to not be pointers but proper bytes
#include "parser.hpp"

#include "sema.lookup.hpp" // Convert lookup AST nodes to references
#include "sema.type_propagator.hpp" // Propigates types through the tree

int main(void) {
	std::string test = R"~(# A resizable list of integers
class Vector(object):
    # Attributes
    items: [int] = None
    size: int = 0

    # Constructor
    def __init__(self:"Vector"):
        self.items = [0]

    # Returns current capacity
    def capacity(self:"Vector") -> int:
        return len(self.items)

    # Increases capacity of vector by one element
    def increase_capacity(self:"Vector") -> int:
        self.items = self.items + [0]
        return self.capacity()

    # Appends one item to end of vector
    def append(self:"Vector", item: int):
        if self.size == self.capacity():
            self.increase_capacity()

        self.items[self.size] = item
        self.size = self.size + 1

# A faster (but more memory-consuming) implementation of vector
class DoublingVector(Vector):
    doubling_limit:int = 16

    # Overriding to do fewer resizes
    def increase_capacity(self:"DoublingVector") -> int:
        if (self.capacity() <= self.doubling_limit // 2):
            self.items = self.items + self.items
        else:
            # If doubling limit has been reached, fall back to
            # standard capacity increases
            self.items = self.items + [0]
        return self.capacity()

vec:Vector = None
num:float = 0.

# Create a vector and populate it with The Numbers
vec = DoublingVector()
for num in [4, 8, 15, 16, 23, 42]:
    vec.append(num)
    print(vec.capacity())

)~";
//     std::string test = R"~(
// # Compute x**y
// def exp(x: int, y: int) -> int:
//     a: int = 0
//     global invocations  

//     def f(i: int) -> int:
//         nonlocal a
//         def geta() -> int:
//             return a
//         if i <= 0:
//             return geta()
//         else:
//             a = a * x
//             return f(i-1)
//     pass
// #    a = 1
// #    invocations = invocations + 1
// #    return f(y)

// invocations:int = 0
// print(exp(2, 10))
// print(exp(3, 3))
// print(invocations) 
//     )~";
	// std::string test = "class x(object):\n\tdef add(a: int, b: int):\n\t\tpass\nif True:\n\tz = x.y()[5] % 5 if X > Y else [5, 5.6, 'h', True, False, None]";
// 	std::string test = R"~(
// class foo(object):
//     def add(x: int, y: int) -> int:
//         return x + y

// x: foo = None
// x = foo()
// x.add(5, 6)
//     )~";

	auto [ast, interner, builtin_block, builtin_size] = initialize_builtin_block();
	auto parser = initialize_parser(ast, interner);

	AST::ref root;
	auto source = preprocess_indentation(test);
	parser.parse(source, root, interner.intern("<generated>").data());
	AST::canonicalize_locations{ast, source}.visit(root);
	ast[builtin_block].as_block().elements.push_back(root);
	ast[root].as_node_base().scope_block = builtin_block;
	if(!diagnostics::singleton().print()) return -1;

    bool changed = true;
    while(changed) {
        changed = sema::resolve_lookups{ast, source}.start(0);
        if(!diagnostics::singleton().print()) return -2;

        sema::type_propagator{ast, interner, source, builtin_size}.visit(0);
        if(!diagnostics::singleton().print()) return -3;
    }

	std::string reconstructed = AST::pretty_printer(ast).visit(builtin_block);
	std::cout << reconstructed << std::endl;

	// std::cout << root << std::endl;

	// parser.enable_ast();
	// std::shared_ptr<peg::Ast> AST;
	// auto preprocessed = preprocess_indentation(test);
	// if(parser.parse(preprocessed, AST)) {
	//     AST = parser.optimize_ast(AST);
	//     std::cout << peg::ast_to_s(AST);
	// }
}