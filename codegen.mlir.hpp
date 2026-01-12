#include "AST.hpp"
#include "AST.print.hpp"
#include "builtin_scope.hpp"
#include "string_helpers.hpp"
#include <cstddef>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <unordered_set>

namespace codegen {

	struct mlir : public AST::visiter<std::string, std::string&> {
		string_interner& interner;
		std::string_view source;
		builtin_scope builtin;

		size_t name_counter = 0;
		size_t indent = 1;

		std::string globals;

		explicit mlir(AST::flattened &ast, string_interner& interner, std::string_view source, builtin_scope builtin)
			: visiter<std::string, std::string&>(ast), interner(interner), source(source), builtin(builtin) {}

		std::string start(AST::ref ref) {
			std::string res = R"~(module {
	// references to builtin functions
	func.func private @allocate$i64s(%count: index) -> memref<?xi8>
	func.func private @as$i64s(%memref: memref<?xi8>) -> memref<?xi64>
	func.func private @as$memrefs(%memref: memref<?xi8>, %offset_bytes: index) -> memref<?xmemref<?xi8>>
	func.func private @assert$size(%memref: memref<?xi8>, %size : index)
	func.func private @assert$i64_count(%memref: memref<?xi8>, %count : index)
	func.func private @__tag__$object(%memref : memref<?xi8>) -> i64
	func.func private @__print__$object(%memref : memref<?xi8>) -> memref<?xi8>
	func.func private @__assert__$int(%memref: memref<?xi8>)
	func.func private @__box__$int(%v: i64) -> memref<?xi8>
	func.func private @__unbox__$int(%memref: memref<?xi8>) -> i64
	func.func private @__print__$int(%memref : memref<?xi8>) -> memref<?xi8>
	func.func private @__print__$float(%memref : memref<?xi8>) -> memref<?xi8>
	func.func private @__print__$str(%memref : memref<?xi8>) -> memref<?xi8>

	func.func private @print(%memref : memref<?xi8>) -> memref<?xi8>

	// globals
)~";
			std::string main;
			visit(ref, main);

			// TODO: This should probably not be manually run...
			auto __print__ = interner.intern("__print__");
			for(auto mem: ast[builtin.object].as_class_declaration().elements)
				if(ast[mem].is_function_declaration())
					if(ast[mem].as_function_declaration().name == __print__) {
						globals += build_dispatcher_function(mem);
						break;
					}

			res += globals;
			res += "\n";
			res += "\tfunc.func @main() -> i32 {\n";
			res += "\n";
			res += "\t\t// main code\n";
			res += main;
			res += "\n";
			res += "\t\t\%out = arith.constant 0 : i32\n";
			res += "\t\treturn \%out : i32\n";
			res += "\t}\n";
			res += "}\n";

			return res;
		}



		std::string gen_indent() { return std::string(indent, '\t'); }

		std::string next_ssa_name() { return "%" + std::to_string(name_counter++); }

		std::string build_dispatcher_function(AST::ref method) {
			assert(ast[method].is_function_declaration());
			if(ast[method].as_function_declaration().overloads != AST::absent)
				method = ast[method].as_function_declaration().overloads;
			auto& func = ast[method].as_function_declaration();
			assert(!func.overloaded_by.empty());

			auto name = std::string(func.name);
			auto object = std::string(ast[func.scope_block].as_class_declaration().name);

			std::string out = "\n";
			out += gen_indent() + "func.func @" + name + "$" + object + "$dispatcher(";
			for(size_t i = 0; i < func.num_parameters - 1; ++i)
				out += "\%arg" + std::to_string(i) + " : memref<?xi8>, ";
			out += "\%arg" + std::to_string(func.num_parameters - 1) + " : memref<?xi8>) -> memref<?xi8> {\n";
			out += gen_indent() + "\t%0 = func.call @__tag__$object(\%arg0) : (memref<?xi8>) -> i64\n";
			out += gen_indent() + "\t\%tag = arith.index_cast %0 : i64 to index\n";
			out += gen_indent() + "\t%1 = scf.index_switch \%tag -> memref<?xi8>\n";

			bool has_default = func.overloaded_by.contains(builtin.object);
			for(auto [type, method]: func.overloaded_by) {
				if(type == builtin.object) continue;
				auto type_name = std::string(ast[type].as_class_declaration().name);

				out += gen_indent() + "\tcase " + std::to_string(type) + " { // " + type_name + "\n";
				out += gen_indent() + "\t\t%2 = func.call @" + name + "$" + type_name + "(";
				for(size_t i = 0; i < func.num_parameters - 1; ++i)
					out += "\%arg" + std::to_string(i) + ", ";
				out += "\%arg" + std::to_string(func.num_parameters - 1) + ") : (";
				for(size_t i = 0; i < func.num_parameters - 1; ++i)
					out += "memref<?xi8>, ";
				out += "memref<?xi8>) -> memref<?xi8>\n";
				out += gen_indent() + "\t\tscf.yield %2 : memref<?xi8>\n";
				out += gen_indent() + "\t}\n";
			}

			if(has_default) {
				std::string type_name = "object";

				out += gen_indent() + "\tdefault { // " + type_name + "\n";
				out += gen_indent() + "\t\t%2 = func.call @" + name + "$" + type_name + "(";
				for(size_t i = 0; i < func.num_parameters - 1; ++i)
					out += "\%arg" + std::to_string(i) + ", ";
				out += "\%arg" + std::to_string(func.num_parameters - 1) + ") : (";
				for(size_t i = 0; i < func.num_parameters - 1; ++i)
					out += "memref<?xi8>, ";
				out += "memref<?xi8>) -> memref<?xi8>\n";
				out += gen_indent() + "\t\tscf.yield %2 : memref<?xi8>\n";
				out += gen_indent() + "\t}\n";
			} else {
				out += gen_indent() + "\tdefault { // error\n";
				out += gen_indent() + "\t\t\%false = arith.constant 0 : i1\n";
				out += gen_indent() + "\t\tcf.assert \%false, \"Type does not support __print__\"\n";
				out += gen_indent() + "\t\t\%one = arith.constant 1 : index\n";
				out += gen_indent() + "\t\t%2 = memref.alloc(\%one) : memref<?xi8>\n";
				out += gen_indent() + "\t\tscf.yield %2 : memref<?xi8>\n";
				out += gen_indent() + "\t}\n";
			}

			out += gen_indent() + "\tfunc.return %1 : memref<?xi8>\n";
			out += gen_indent() + "}\n";
			return out;
		}

		std::string memref_to_unsized(std::string memref, std::string in_ssa, std::string& out) {
			if(memref == "memref<?xi8>") return in_ssa;

			size_t size;
			std::istringstream str(memref.substr(7));
			str >> size;

			auto ssa = next_ssa_name();
			out += gen_indent() + ssa + " = memref.cast " + in_ssa + " : memref<" + std::to_string(size) + "xi8> to memref<?xi8>\n";
			return ssa;
		}




		std::string visit_ref(AST::ref& target, AST::ref r, std::string& out) override {
			// changed = true;
			// return target = visit(target);
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_type_lookup(AST::type_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_list_type(AST::list_type& value, AST::ref r, std::string& out) override {
			// as_type_block([&](){ value.type = visit(value.type); });
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_function_type(AST::function_type& value, AST::ref r, std::string& out) override {
			// return as_type_block([&](){
			// 	for(auto& arg: value.elements)
			// 		arg = visit(arg);
			// 	value.return_type = visit(value.return_type);
			// 	return r;
			// });
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_class_declaration_lookup(AST::class_declaration_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_class_declaration(AST::class_declaration& value, AST::ref r, std::string& out) override {
			// if(as_type) return r;

			// auto level = symbol_table.try_insert(value.name, r);
			// if(level == 0)
			// 	diagnostics::singleton().push_error("Attempting to redeclare class", source, value.location);
			// // if(level != AST::absent) // TODO: crashes?
			// // 	diagnostics::singleton().push_warning("Class shadows another symbol", source, value.location);
			// return visit_block(value, r);
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_function_declaration(AST::function_declaration& value, AST::ref r, std::string& out) override {
			// auto level = symbol_table.try_insert(value.name, r);
			// if(level == 0)
			// 	diagnostics::singleton().push_error("Attempting to redeclare function", source, value.location);
			// if(level != AST::absent)
			// 	diagnostics::singleton().push_warning("Function shadows another symbol", source, value.location);
			// if(value.return_type != AST::absent) as_type_block([&](){ value.return_type = visit(value.return_type); });
			// return visit_block(value, r);
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_parameter_declaration(AST::parameter_declaration& value, AST::ref r, std::string& out) override {
			// auto level = symbol_table.try_insert(value.name, r);
			// if(level == 0)
			// 	diagnostics::singleton().push_error("Attempting to redeclare parameter", source, value.location);
			// // if(level != AST::absent) // TODO: crashes!
			// // 	diagnostics::singleton().push_warning("Parameter shadows another symbol", source, value.location);
			// if(value.type != AST::absent) as_type_block([&](){ value.type = visit(value.type); });
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_variable_declaration(AST::variable_declaration& value, AST::ref r, std::string& out) override {
			// auto level = symbol_table.try_insert(value.name, r);
			// if(level == 0)
			// 	diagnostics::singleton().push_error("Attempting to redeclare variable", source, value.location);
			// if(level != AST::absent)
			// 	diagnostics::singleton().push_warning("Variable shadows another symbol", source, value.location);
			// as_type_block([&](){ value.type = visit(value.type); });
			// value.initial_value = visit(value.initial_value);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_global_lookup(AST::global_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_global(AST::global& value, AST::ref r, std::string& out) override {
			// if(!is_declaration(value.reference))
			// 	diagnostics::singleton().push_error("Global does not point to a declaration", source, value.location);

			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_nonlocal_lookup(AST::nonlocal_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_nonlocal(AST::nonlocal& value, AST::ref r, std::string& out) override {
			// if(!is_declaration(value.reference))
			// 	diagnostics::singleton().push_error("Nonlocal does not point to a declaration", source, value.location);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_pass_statement(AST::pass_statement&, AST::ref r, std::string& out) override {
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_return_statement(AST::return_statement& value, AST::ref r, std::string& out) override {
			// if(value.what != AST::absent) value.what = visit(value.what);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_assignment(AST::assignment& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_if_statement(AST::if_statement& value, AST::ref r, std::string& out) override {
			// for(auto& [condition, block]: value.condition_block_pairs) {
			// 	if(condition != AST::absent) condition = visit(condition);
			// 	visit_block(block, r);
			// }
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_while_statement(AST::while_statement& value, AST::ref r, std::string& out) override {
			// value.condition = visit(value.condition);
			// return visit_block(value, r);
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_for_statement_lookup(AST::for_statement_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_for_statement(AST::for_statement& value, AST::ref r, std::string& out) override {
			// value.source = visit(value.source);
			// return visit_block(value, r);
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_block(AST::block& value, AST::ref r, std::string& out) override {
			++indent;
			for(auto elem: value.elements)
				visit(elem, out);
			--indent;
			return "";
		}

		std::string visit_if_expression(AST::if_expression& value, AST::ref r, std::string& out) override {
			// value.then = visit(value.then);
			// value.condition = visit(value.condition);
			// value.else_ = visit(value.else_);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_explicit_cast(AST::explicit_cast& value, AST::ref r, std::string& out) override {
			// value.reference = visit(value.reference);
			// value.type = visit(value.type);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_logical_and(AST::logical_and& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_logical_or(AST::logical_or& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_equal(AST::equal& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_not_equal(AST::not_equal& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_less(AST::less& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_less_equal(AST::less_equal& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_greater(AST::greater& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_greater_equal(AST::greater_equal& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_is(AST::is& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_add(AST::add& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_subtract(AST::subtract& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_multiply(AST::multiply& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_quotient(AST::quotient& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_remainder(AST::remainder& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_divide(AST::divide& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_logical_not(AST::logical_not& value, AST::ref r, std::string& out) override {
			// value.what = visit(value.what);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_negate(AST::negate& value, AST::ref r, std::string& out) override {
			// value.what = visit(value.what);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_variable_load_lookup(AST::variable_load_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_variable_load(AST::variable_load& value, AST::ref r, std::string& out) override {
			if(ast[value.reference].is_function_declaration()) {
				return std::string{ast[value.reference].as_function_declaration().name};
			}

			throw std::runtime_error("Not yet implemented");
			// if(value.reference == AST::absent)
			// 	diagnostics::singleton().push_error("Attempting to access non-existant variable", source, value.location);
			// return r;
		}

		std::string visit_variable_store_lookup(AST::variable_store_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_variable_store(AST::variable_store& value, AST::ref r, std::string& out) override {
			// if(value.reference == AST::absent)
			// 	throw std::runtime_error("Invalid variable store");

			// if(ast[value.reference].is_function_declaration()) {
			// 	diagnostics::singleton().push_error("Attempting to assign to a function", source, value.location);
			// 	return r;
			// }
			// if(ast[value.reference].is_class_declaration() || ast[value.reference].is_class_declaration_lookup()) {
			// 	diagnostics::singleton().push_error("Attempting to assign to a class", source, value.location);
			// 	return r;
			// }

			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_member_access_lookup(AST::member_access_lookup& value, AST::ref r, std::string& out) override {
			throw std::runtime_error("Lookups should already be resolved");
		}

		std::string visit_member_access(AST::member_access& value, AST::ref r, std::string& out) override {
			// // TODO: Should we do any validation here?
			// visit(value.lhs);
			// // visit(value.reference); // creates cycle!
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_array_index(AST::array_index& value, AST::ref r, std::string& out) override {
			// value.lhs = visit(value.lhs);
			// value.rhs = visit(value.rhs);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_call(AST::call& value, AST::ref r, std::string& out) override {
			auto lhs_type = ast[value.lhs].as_expression().type;
			auto ssa = next_ssa_name();
			std::string tmp = ""; // We do our modification to a temporary value so that arguments happen before us
			if(ast[lhs_type].is_function_type()) { // Direct call
				auto& type = ast[lhs_type].as_function_type();
				bool is_method = value.elements.size() != type.elements.size(); // TODO: Do something with this!

				auto name = visit(value.lhs, out);
				tmp = gen_indent() + ssa + " = func.call @" + name + "(";

				for(size_t i = 0; i < value.elements.size() - 1; ++i)
					tmp += visit(value.elements[i], out) + ", ";
				if(!value.elements.empty()) tmp += visit(value.elements.back(), out);

				tmp += ") : (";

				for(size_t i = 0; i < value.elements.size() - 1; ++i)
					tmp += "memref<?xi8>, ";
				if(!value.elements.empty()) tmp += "memref<?xi8>";

				tmp += ") -> memref<?xi8>\n";
			} else { // Indirect call
				throw std::runtime_error("Not implemented yet!");
			}
			// value.lhs = visit(value.lhs);
			// for(auto& arg: value.elements)
			// 	arg = visit(arg);
			// return r;
			out += tmp;
			return ssa;
		}

		std::string visit_float_literal(AST::float_literal& value, AST::ref r, std::string& out) override {
			// as_type_block([&](){ value.type = visit(value.type); }); // Lookup type!
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_int_literal(AST::int_literal& value, AST::ref r, std::string& out) override {
			// as_type_block([&](){ value.type = visit(value.type); }); // Lookup type!
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_string_literal(AST::string_literal& value, AST::ref r, std::string& out) override {
			// Pad to a multiple of 8 bytes
			std::vector<uint8_t> data(value.value.begin(), value.value.end());
			if(data.back() != 0) data.push_back(0); // Make sure the string is null terminated!
			if (data.size() % 8 != 0) {
				size_t pad = 8 - (data.size() % 8);
				data.insert(data.end(), pad, 0);
			}

			std::array<uint64_t, 2> prototype = {builtin.Str, value.value.size()};
			std::span<uint8_t> prototype_bytes{(uint8_t*)prototype.data(), prototype.size() * sizeof(prototype[0])};

			size_t numi8s = prototype_bytes.size() + data.size();
			static std::unordered_set<AST::ref> lits_in_global = {};

			if(!lits_in_global.contains(r)) {
				std::ostringstream g;
				g << "\tmemref.global \"private\" constant @lit$" << r
					<< " : memref<" << numi8s <<"xi8> = dense<[";
				for(auto byte: prototype_bytes)
					g << "0x" << std::hex << (int)byte << ", ";
				for(size_t i = 0; i < data.size() - 1; ++i)
					g << "0x" << std::hex << (int)data[i] << ", ";
				g << "0x" << std::hex << (int)data.back() << "]>\n";

				globals += g.str();
				lits_in_global.insert(r);
			}

			auto ssa = next_ssa_name();
			auto memref_type = "memref<" + std::to_string(numi8s) + "xi8>";
			out += gen_indent() + ssa + " = memref.get_global @lit$" + std::to_string(r) + " : " + memref_type + "\n";
			return memref_to_unsized(memref_type, ssa, out);
		}

		std::string visit_bool_literal(AST::bool_literal& value, AST::ref r, std::string& out) override {
			// as_type_block([&](){ value.type = visit(value.type); }); // Lookup type!
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_none_literal(AST::none_literal& value, AST::ref r, std::string& out) override {
			// as_type_block([&](){ value.type = visit(value.type); }); // Lookup type!
			// return r;
			throw std::runtime_error("Not yet implemented");
		}

		std::string visit_list_literal(AST::list_literal& value, AST::ref r, std::string& out) override {
			// for(auto& elem: value.elements)
			// 	elem = visit(elem);
			// return r;
			throw std::runtime_error("Not yet implemented");
		}
	};

} // namespace AST