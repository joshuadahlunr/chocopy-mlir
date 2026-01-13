#pragma once

#include <any>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "diagnostics.hpp"
#include "string_helpers.hpp"

namespace AST {
	namespace detail {
		template <typename T, typename Variant>
		struct variant_index;

		template <typename T, typename... Types>
		struct variant_index<T, std::variant<Types...>> {
			template <std::size_t I>
			static constexpr std::size_t impl() {
				using Current = std::tuple_element_t<I, std::tuple<Types...>>;
				if constexpr (std::is_same_v<T, Current>)
					return I;
				else return impl<I + 1>();
			}

			static constexpr std::size_t value = impl<0>();
		};

		template <typename T, typename Variant>
		inline constexpr std::size_t variant_index_v = variant_index<T, Variant>::value;

	}

	using ref = size_t;
	static constexpr ref absent = -1;
	inline ref a2r(const std::any& any) {
		return std::any_cast<ref>(any);
	}

	struct node_base {
		diagnostics::source_location location;
		ref scope_block = absent;
	};

	struct error : public node_base {};

	struct lookup {
		interned_string interned_name;
	};
	struct referenced {
		ref reference;
	};
	struct type_lookup : public node_base, public lookup { };
	// TODO: Non lookup versions
	struct list_type : public node_base {
		ref type;
	};
	struct ref_list {
		std::vector<ref> elements;
	};
	struct function_type: public node_base, public ref_list {
		ref return_type = absent;
	};

	

	struct block: public node_base, public ref_list {};


	struct class_declaration_lookup : public block {
		interned_string name;
		interned_string base;
	};
	struct class_declaration : public block {
		interned_string name;
		ref base;
		size_t size, alignment;
	};

	struct function_declaration : public block {
		interned_string name;
		ref return_type;
		size_t num_parameters;

		ref overloads = absent;
		std::unordered_map<ref, ref> overloaded_by; // maps type -> method
	};
	struct parameter_declaration : public node_base {
		interned_string name;
		ref type;
		size_t index;
	};

	struct variable_declaration : public node_base {
		interned_string name;
		ref type;
		ref initial_value;
	};

	struct global_lookup : public node_base, public lookup {};
	struct global : public node_base, public referenced {};
	struct nonlocal_lookup : public node_base, public lookup {};
	struct nonlocal : public node_base, public referenced {};



	struct pass_statement : public node_base {};
	struct return_statement : public node_base {
		ref what = absent;
	};

	struct expression : public node_base {
		ref type = absent;
	};
	struct binary_op : public expression {
		ref lhs, rhs;
	};
	struct assignment: public binary_op {}; // I am modeling assignment as an expression... to allow chaining

	struct if_statement : public node_base {
		struct condition_block {
			ref condition = absent;
			struct block block;
		};
		std::vector<condition_block> condition_block_pairs; // else is last and has condition -1
	};

	struct while_statement: public block {
		ref condition;
	};
	struct for_statement_lookup: public block, public lookup {
		ref source;
	};
	struct for_statement: public block, public referenced {
		ref source;
	};



	struct if_expression : public expression {
		ref then, condition, else_;
	};
	struct explicit_cast : public expression, public referenced {};

	struct logical_and : public binary_op {};
	struct logical_or : public binary_op {};

	struct equal : public binary_op {};
	struct not_equal : public binary_op {};
	struct less : public binary_op {};
	struct less_equal : public binary_op {};
	struct greater : public binary_op {};
	struct greater_equal : public binary_op {};
	struct is : public binary_op {};

	struct add : public binary_op {};
	struct subtract : public binary_op {};

	struct multiply : public binary_op {};
	struct quotient : public binary_op {};
	struct remainder : public binary_op {};
	struct divide : public binary_op {};

	struct unary_op : public expression {
		ref what;
	};

	struct logical_not : public unary_op {};
	struct negate : public unary_op {};



	struct list_literal: public expression, public ref_list {};

	struct variable_load_lookup: public expression, public lookup {};
	struct variable_load: public expression, public referenced {};
	struct variable_store_lookup: public expression, public lookup {};
	struct variable_store: public expression, public referenced {};

	struct member_access_lookup: public expression, public lookup {
		ref lhs, resolve_type = absent;
	};
	struct member_access: public expression, public referenced {
		ref lhs;
	};

	struct array_index : public expression {
		ref lhs, rhs;
	};

	struct call: public expression, public ref_list {
		ref lhs;
	};

	struct bool_literal : public expression {
		bool value;
	};
	struct string_literal : public expression {
		interned_string value;
	};
	struct int_literal : public expression {
		int32_t value;
	};
	struct float_literal : public expression {
		float value;
	};
	struct none_literal : public expression {}; // None literal


	#define NODE_TYPES_STAMPER(X) \
		X(ref)\
\
		X(type_lookup)\
		X(list_type)\
		X(function_type)\
\
		/* declarations */\
		X(class_declaration_lookup)\
		X(class_declaration)\
		X(function_declaration)\
		X(parameter_declaration)\
		X(variable_declaration)\
		X(global_lookup)\
		X(global)\
		X(nonlocal_lookup)\
		X(nonlocal)\
\
		/* statements */\
		X(pass_statement)\
		X(return_statement)\
		X(assignment)\
		X(if_statement)\
		X(while_statement)\
		X(for_statement_lookup)\
		X(for_statement)\
		X(block)\
\
		X(if_expression)\
		X(explicit_cast)\
\
		/* binary */\
		X(logical_and)\
		X(logical_or)\
		X(equal)\
		X(not_equal)\
		X(less)\
		X(less_equal)\
		X(greater)\
		X(greater_equal)\
		X(is)\
		X(add)\
		X(subtract)\
		X(multiply)\
		X(quotient)\
		X(remainder)\
		X(divide)\
\
		/* unary */\
		X(logical_not)\
		X(negate)\
\
		/* postfix */\
		X(variable_load_lookup)\
		X(variable_load)\
		X(variable_store_lookup) /* target */\
		X(variable_store)\
		X(member_access_lookup)\
		X(member_access)\
		X(array_index)\
		X(call)\
\
		/* Literals */\
		X(float_literal)\
		X(int_literal)\
		X(string_literal)\
		X(bool_literal)\
		X(none_literal)\
		X(list_literal)

	#define APPEND_COMMA(x) x ,
	struct node: public std::variant<NODE_TYPES_STAMPER(APPEND_COMMA) error> {
		using variant = std::variant<NODE_TYPES_STAMPER(APPEND_COMMA) error>;

#define IMPLEMENT_HELPERS(type)\
		bool is_##type() { return std::holds_alternative<type>(*this); }\
		type& as_##type() { return std::get<type>(*this); }\
		const type& as_##type() const { return std::get<type>(*this); }

		bool is_error() { return std::holds_alternative<error>(*this); }

		node_base& as_node_base() {
			node_base* out = nullptr;
			std::visit([&out](auto& a) {
				out = &(node_base&)a;
			}, *this);
			assert(out != nullptr);
			return *out;
		}

		expression& as_expression() {
			expression* out = nullptr;
			std::visit([&out](auto& a) {
				out = &(expression&)a;
			}, *this);
			assert(out != nullptr);
			return *out;
		}

		NODE_TYPES_STAMPER(IMPLEMENT_HELPERS)
	};

	using flattened = std::vector<node>;

	template<typename... Ts>
	ref make_node(flattened& AST, Ts... args) {
		ref out = AST.size();
		AST.emplace_back(std::forward<Ts>(args)...);
		return out;
	}

	inline ref make_error(flattened& AST, const diagnostics::source_location& location) {
		ref out = AST.size();
		AST.emplace_back(error{location});
		return out;
	}

	template<typename Treturn>
	struct visiter {
		AST::flattened& ast;

		visiter(AST::flattened& ast) : ast(ast) {}

#define IMPLEMENT_VISIT_DECL(type)\
		virtual Treturn visit_##type(type& value, AST::ref ref) = 0;

		NODE_TYPES_STAMPER(IMPLEMENT_VISIT_DECL)

		Treturn visit(node& n, AST::ref r) {
			switch(n.index()) {
#define IMPLEMENT_VISIT(type)\
				case detail::variant_index_v<type, node::variant>:\
					return visit_##type(std::get<type>(n), r);

				NODE_TYPES_STAMPER(IMPLEMENT_VISIT)

				case detail::variant_index_v<error, node::variant>:
					// Do nothing on errors...
					if constexpr(std::is_same_v<Treturn, void>)
						return;
					else return {};
				default:
					throw std::runtime_error("Invalid node state");
			}

		}

		Treturn visit(const ref ref) {
			return visit(ast[ref], ref);
		}
	};
}