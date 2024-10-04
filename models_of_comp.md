---
author: Dario Loi
title: Computer Science Course Notes --- 1st Semester
---


# Models of Computation

## First Lesson

### Contact Information

Prof email: piperno@di.uniroma1.it

*Actual* lecture times:

- Wednsday 13:30 - 15:00
- Thursday 16:00 - 17:30, or 16:15 - 17:55

### Course Contents

The course content will focus on **functional programming** and $\lambda$-calculus.

The main application of these languages is that of *function application*, these languages
are devoid of *assignment* semantics, and are therefore called *pure*.

The main ingredients of our programs will be:

- Variables, which are usually denoted by lowercase letters
- Starting from these, you obtain the set of all programs, which are called $\lambda$-terms,
this set is denoted as $\Lambda$.

### Set Definition

We can provide an inductive definition of a set of $\lambda$-terms, $\Lambda$:

$$\frac{x \in V}{x \in \Lambda} \quad \text{(var)}$$

$$
 \frac{M \in \Lambda \quad N \in \Lambda}{(M N) \in \Lambda} \quad \text{(app)}
$$

$$
    \frac{M \in \Lambda \quad x \in V}{\lambda x . M \in \Lambda} \quad \text{(abs)}
$$

Applying these inductive rules (variables, application, abstraction).

### Bactus Normal Form

Another way of describing a set of lambda terms is by using Bactus Normal Form (BNF).

$$
\Lambda :: Var | \Lambda \Lambda | \lambda Var . \Lambda
$$

Which is essentially a grammar for the set of lambda terms.

Lambda calculus is left-associative:

$$
((x y) z) = x y z \neq x (y z)
$$

All functions are unary (we assume currying). For example, the function $f(x, y)$ is represented as: $$\lambda x . \lambda y . f x y$$.

Functions in lambda calculus can be applied to other functions or themselves, they can also return functions.

### Beta Reduction

The main operation in lambda calculus is *beta reduction*, which is the application of a function to an argument.

$$
\underset{redex}{\underline{(\lambda x . M) N}} \rightarrow_\beta M[N/x]
$$

Where $M[N/x]$ is the result of substituting all occurrences of $x$ in $M$ with $N$.

Some other examples:

$$
(\lambda x.x) y \rightarrow_\beta y
$$

$$
(\lambda x . x x) y \rightarrow_\beta y y
$$

$$
(\lambda x y . y x) (\lambda u . u) \rightarrow_\beta \lambda y . y (\lambda u . u)
$$


$$
  (\lambda x y . y x) ( \lambda t . y) \rightarrow_\beta \lambda y . y (\lambda t . y)
$$

This rule can be applied in any context in which it appears.

#### Formal Substitution Definition

We give a formal definition of substitution, $M[N/x]$:

$$
x[N/x] = N
$$

$$
y[N/x] = y
$$

$$
(M_1 M_2)[N/x] = M_1[N/x] M_2[N/x]
$$

$$
(\lambda t . P)[N/x] = \lambda t . (P[N/x])
$$

As observed, substitution is always in place of *free* variables, therefore the abstraction is *not* replaced in the last rule.

If we had an abstraction of type $\lambda x . P$ where $x \in P$, it would be best to rename $x$ in order to avoid name clashes.

### Types of Variables

We distinguish two kinds of variables:

- Free variables: variables that are not bound by an abstraction
- Bound variables: variables that are bound by an abstraction

For example, in the term $\lambda x . x y$, $y$ is a free variable, while $x$ is a bound variable.

Bound variables can be renamed, whereas for free variables the naming is relevant.

$$
\begin{cases}
FV(x) = \left\{x\right\}\\
FV(MN) = FV(M) \cup FV(N)\\
FV(\lambda x . M) = FV(M) - \left\{ x \right\}
\end{cases}
$$

A set of lambda terms where $LM(\Lambda) = \empty$ is called *closed*.

### Extra Rules


$$(\mu) \quad \frac{M \rightarrow_\beta M^\prime}{NM \rightarrow_\beta NM^\prime}$$
$$(\nu) \quad \frac{M \rightarrow_\beta M^\prime}{MN \rightarrow_\beta M^\prime N}$$
$$(\xi) \quad \frac{M \rightarrow M^\prime}{\lambda x . M \rightarrow \lambda x . M^{\prime}}$$


These rules allow us to select redexes in a context-free manner in the middle of our lambda term. We can then choose the order of evaluation of our redexes, while still taking care of the left-associative order of precedence. Our calculus is therefore not *determinate* but is still *deterministic*, meaning that there may be multiple reduction strategies but they all lead to the same result.

This corollary is called the *Church Rosser Theorem*, discovered in 1936.

In general, a call-by-value-like semantic is preferrable when choosing evaluation paths, as it clears the most amount of terms as early as possible.

* Call By Value is *efficient*
* Call By Name is *complete*, **if** the lambda term is normalizable

## Second Lecture

### Alpha Reduction

Alpha reduction is the renaming of bound variables in a lambda term.

$$
\lambda x . M \rightarrow_\alpha \lambda y . M[y/x]
$$

This rule is used to avoid name clashes between bound variables.

### Arithmetic Expressions

The set of all valid arithmetic expressions has a very precise syntax. In general, a syntax can be viewed either as a tool for checking validity or as a generator of valid expressions (a grammar).

We proceed to give a definition for arithmetic expressions

$$
\frac{x \in \mathbb{N}}{x \in \text{Expr}} \quad \text{(num)}
$$

$$
\frac{X \in \text{Expr} \quad Y \in \text{Expr}}{X + Y \in \text{Expr}} \quad \text{(add)}
$$

$$
\frac{X \in \text{Expr} \quad Y \in \text{Expr}}{X \times Y \in \text{Expr}} \quad \text{(mul)}
$$
Etc, etc... for all the other binary operations.

From this, we can successfully decompose any arithmetic expression into a syntactic tree.
With this set of rules, we have a slight problem: we can't represent negative numbers. We could solve this either by adding a rule for unary minus, or by specifying the num rule over $\mathbb{Z}$ instead of $\mathbb{N}$.

### Combinators

We define three combinators:

$$
S = \lambda x y z. x z (y z)
$$

$$
K = \lambda x y . x
$$

$$
I = \lambda u . u
$$

We have that $SK y \to_\beta I$

*Exercise 1.1:* $\beta$-reduce $S (KS) S$

This reduces to $\lambda z b c . z (b c)$, which is the $B$ combinator (composition).

*Exercise 1.2:* $\beta$-reduce $S (BBS) (KK)$. 

This reduces to $\lambda z c d . z d c$, which is the $C$ combinator (permutation).

For exercise this, we show a step-by-step reduction:

$$
\begin{aligned}
&S(BBS)(KK)\\
\leadsto & \lambda z . (BBS) z ((KK) z)\\
\leadsto & \lambda z . (\lambda c . B(Sc)) z ((KK) z)\\
\leadsto & \lambda z . (\lambda c . B(Sc)) z K\\
\leadsto & \lambda z . B(Sz) K\\
\leadsto & \lambda z . (\lambda c . (Sz) (Kc)\\
\leadsto & \lambda z c . Sz (Kc)\\
\leadsto & \lambda z c d . zd (Kcd)\\
\leadsto & \lambda z c d . zd c \quad \square.
\end{aligned}
$$

## Third Lecture

    Recupera!!!

## Fourth Lecture

First we do a simple exercise, $\beta$-reduce $\lambda u v. ( \lambda z . z z) (\lambda t . t u v).$ 

$$
\begin{aligned}
&\lambda u v. ( \lambda z . z z) (\lambda t . t u v)\\
\rightarrow_\beta & \lambda u v. (\lambda t . t u v) (\lambda t . t u v)\\
\rightarrow_\beta & (\lambda t . t u v) (\lambda t . t u v)\\
\rightarrow_\beta & \lambda u v . (\lambda t. t u v) u v\\
\rightarrow_\beta & \lambda u v . u u v v \quad \square.
\end{aligned}
$$

Find a term $X$ s.t $Xx = \lambda t . t ( X x)$

$$
\begin{aligned}
&X x = \lambda t . t (X x)\\
&X = (\lambda f x y . t (f x)) X\\
&X = Y (\lambda f x y . t (f x))\\
\end{aligned}
$$

Find a term $H$ s.t $H (\lambda x_1 x_2 x_3.P) = \lambda a x_3 x_2 x_1 . a x_1 x_2 x_3$

