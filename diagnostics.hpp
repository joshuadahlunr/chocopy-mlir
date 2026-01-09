
#include "string_helpers.hpp"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <source_location>
#include <sstream>
#include <stack>
#include <string>
#include <string_view>

struct diagnostics {
	static diagnostics& singleton() {
		static diagnostics diagnostics;
		return diagnostics;
	}

	std::stack<std::string> errors;
	std::stack<std::string> warnings;

	struct source_location {
		size_t start_byte, end_byte;
		std::string_view filename;

		struct line_column {
			size_t line, column;
		};
		static line_column find_line_column(size_t byte, const std::string_view source) {
			struct line_column out = {0, 0};
			assert(byte < source.size());
			for(size_t i = 0; i < byte; ++i) {
				++out.column;
				if(source[i] == '\n' || (source[i] == '\r' && source.size() > i + 1 && source[i + 1] != '\n'))
					++out.line, out.column = 0;
			}
			return out;
		}

		line_column start_line_column(const std::string_view source) const {
			return find_line_column(start_byte, source);
		}
		size_t start_line(const std::string_view source) const {
			return start_line_column(source).line;
		}
		size_t start_column(const std::string_view source) const {
			return start_line_column(source).column;
		}

		line_column end_line_column(const std::string_view source) const {
			return find_line_column(end_byte, source);
		}
		size_t end_line(const std::string_view source) const {
			return end_line_column(source).line;
		}
		size_t end_column(const std::string_view source) const { 
			return end_line_column(source).column;
		}

		source_location& pointers_to_bytes(const std::string_view source) {
			start_byte = (char *)start_byte - source.data();
			end_byte = (char *)end_byte - source.data();
			return *this;
		}
	};

	static std::string generate_diagnostic(std::string_view type, std::string_view hint, std::string_view source_code, source_location location) {
		auto start = location.start_line_column(source_code), end = location.end_line_column(source_code);
		auto lines = split(source_code, '\n');
		std::ostringstream out;
		
		size_t remaining = std::max<size_t>(80 - (end.column - start.column), 0);
		size_t start_column = std::max<int64_t>(int64_t(start.column) - remaining / 2, 0);
		for(size_t i = start.line; i <= end.line; ++i) {
			auto line = lines[i];
			size_t end_column = std::min<size_t>(end.column + remaining / 2, line.size() - 1);
			size_t spacer = start.column - start_column;
			
			out << type << " - " << location.filename << ":" << (start.line + 1) << ":" << (start.column + 1) << "\n"
				<< line.substr(start_column, end_column - start_column) << "\n"
				<< std::string(spacer, ' ') << std::string(end.column - start.column, '^') << "\n"
				<< std::string(spacer, ' ') << hint;
			if(i < end.line) out << "\n";
		}
		return out.str();
	}

	source_location push_error(std::string_view hint, std::string_view source_code, source_location location, std::string_view type = "Error") {
		errors.push(generate_diagnostic(type, hint, source_code, location));
		return location;
	}
	source_location push_warning(std::string_view hint, std::string_view source_code, source_location location, std::string_view type = "Warning") {
		warnings.push(generate_diagnostic(type, hint, source_code, location));
		return location;
	}

	bool should_continue() {
		return errors.empty();
	}

	bool print(std::ostream& out = std::cerr) {
		bool should_continue = this->should_continue();

		while(!errors.empty()) {
			out << errors.top() << std::endl;
			errors.pop();
		}

		while(!warnings.empty()) {
			out << warnings.top() << std::endl;
			warnings.pop();
		}

		return should_continue;
	}
};