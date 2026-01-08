#pragma once

#include <any>
#include <cstddef>
#include <stdexcept>
#include <variant>
#include <vector>

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

	struct lookup {
		interned_string interned_name;
	};
	struct referenced {
		ref reference;
	};
	struct type_lookup : public lookup { };
	// TODO: Non lookup versions
	struct list_type {
		ref type;
	};

	struct ref_list {
		std::vector<ref> elements;
	};
	struct block: public ref_list {};



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
	};
	struct parameter_declaration {
		interned_string name;
		ref type;
		size_t index;
	};

	struct variable_declaration {
		interned_string name;
		ref type;
		ref initial_value;
	};

	struct global_lookup : public lookup {};
	struct global : public referenced {};
	struct nonlocal_lookup : public lookup {};
	struct nonlocal : public referenced {};



	struct pass_statement {};
	struct return_statement {
		ref what = absent;
	};

	struct binary_op {
		ref lhs, rhs;
	};
	struct assignment: public binary_op {}; // I am modeling assignment as an expression... to allow chaining

	struct if_statement {
		struct condition_block {
			ref condition = absent;
			struct block block;
		};
		std::vector<condition_block> condition_block_pairs; // else is last and has condition -1
	};

	struct while_statement: public block {
		ref condition;
	};
	struct for_statement: public block {
		interned_string iterator; 
		ref source;
	};



	struct if_expression {
		ref then, condition, else_;
	};

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

	struct unary_op {
		ref what;
	};

	struct logical_not : public unary_op {};
	struct negate : public unary_op {};



	struct list_literal: public ref_list {};

	struct variable_load_lookup: public lookup {};
	struct variable_load: public referenced {};
	struct variable_store_lookup: public lookup {};
	struct variable_store: public referenced {};

	struct member_access_lookup: public lookup {
		ref lhs;
	};
	struct member_access: public referenced {
		ref lhs;
	};

	struct array_index {
		ref lhs, rhs;
	};

	struct call: ref_list {
		ref lhs;
	};


	struct none {}; // None literal


	#define NODE_TYPES_STAMPER(X) \
		X(ref)\
\
		X(type_lookup)\
		X(list_type)\
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
		X(for_statement)\
		X(block)\
\
		X(if_expression)\
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
		X(double)\
		X(interned_string)\
		X(bool)\
		X(none)\
		X(list_literal)

	#define APPEND_COMMA(x) x ,

	struct unused_spacer {};
	struct node: public std::variant<NODE_TYPES_STAMPER(APPEND_COMMA) unused_spacer> {
		using variant = std::variant<NODE_TYPES_STAMPER(APPEND_COMMA) unused_spacer>;

#define IMPLEMENT_HELPERS(type)\
		bool is_##type() { return std::holds_alternative<type>(*this); }\
		type& as_##type() { return std::get<type>(*this); }\
		const type& as_##type() const { return std::get<type>(*this); }

		NODE_TYPES_STAMPER(IMPLEMENT_HELPERS)
	};

	using flattened = std::vector<node>;

	template<typename... Ts>
	ref make_node(flattened& AST, Ts... args) {
		ref out = AST.size();
		AST.emplace_back(std::forward<Ts>(args)...);
		return out;
	}

	template<typename Treturn>
	struct visiter {
#define IMPLEMENT_VISIT_DECL(type)\
		virtual Treturn visit_##type(const type&, AST::ref ref) = 0;

		NODE_TYPES_STAMPER(IMPLEMENT_VISIT_DECL)

		Treturn visit(const node& n, AST::ref r) {
			switch(n.index()) {
#define IMPLEMENT_VISIT(type)\
				case detail::variant_index_v<type, node::variant>:\
					return visit_##type(std::get<type>(n), r);

				NODE_TYPES_STAMPER(IMPLEMENT_VISIT)

				default: throw std::runtime_error("Invalid node state");
			}
		}
	};
}