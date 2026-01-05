
#include <algorithm>
#include <iterator>
#include <peglib.h>
#include <assert.h>
#include <iostream>

#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>

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

std::string preprocess_indentation(const std::string& src) {
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
nums = [4, 8, 15, 16, 23, 42]
i = 0
while i < len(nums):
    vec.append(nums[i])
    print(vec.capacity())
    i = i + 1
    
)~";
    // std::string test = "for x in true:\n\tpass";

	auto grammar =
#include "chocopy.peg"
	;

	peg::parser parser;
	parser.set_logger([](size_t line, size_t col, const std::string& msg, const std::string &rule) {
		std::cerr << line << ":" << col << ": " << msg << std::endl;
	});
	auto ok = parser.load_grammar(grammar);
	assert(ok);

	parser.enable_packrat_parsing(); // Enable packrat parsing.
    // parser.enable_trace([](const peg::Ope &name, const char *s, size_t n, const peg::SemanticValues &vs,
    // const peg::Context &c, const std::any &dt, std::any &trace_data){
    //     std::cout << "> " << vs.name() << std::endl;
    // }, [](const peg::Ope &ope, const char *s, size_t n, const peg::SemanticValues &vs,
    // const peg::Context &c, const std::any &dt, size_t, std::any &trace_data) {
    //     std::cout << "< " << vs.name() << std::endl;
    // });



	int val;
	auto preprocessed = preprocess_indentation(test);
	parser.parse(preprocessed, val);

    // parser.enable_ast();
    // std::shared_ptr<peg::Ast> ast;
	// auto preprocessed = preprocess_indentation(test);
	// if(parser.parse(preprocessed, ast)) {
    //     ast = parser.optimize_ast(ast);
    //     std::cout << peg::ast_to_s(ast);
    // }
}