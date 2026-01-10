
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
num:int = 0

# Create a vector and populate it with The Numbers
vec = DoublingVector()
for num in [4, 8, 15, 16, 23, 42]:
    vec.append(num)
    print(vec.capacity())

)~";

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