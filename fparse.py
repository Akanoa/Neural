"""

http://pyparsing.wikispaces.com/file/view/fourFn.py/30154950/fourFn.py
Copyright 2003-2006 by Paul McGuire

This is a modified version of fourFn.py by Paul McGuire

C.Perreau
"""

from pyparsing import Literal, CaselessLiteral, Word, Combine, Group, Optional, \
    ZeroOrMore, Forward, nums, alphas
import math
import operator
import logging

exprStack = []


def pushFirst(strg, loc, toks):
    exprStack.append(toks[0])


def pushUMinus(strg, loc, toks):
    if toks and toks[0] == '-':
        exprStack.append('unary -')
        # ~ exprStack.append( '-1' )
        #~ exprStack.append( '*' )


bnf = None


def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        point = Literal(".")
        e = Literal("E")
        fnumber = Combine(Word("+-" + nums, nums) +
                          Optional(point + Optional(Word(nums))) +
                          Optional(e + Word("+-" + nums, nums)))
        ident = Word(alphas, alphas + nums + "_$")
        plus = Literal("+")
        minus = Literal("-")
        mult = Literal("*")
        div = Literal("/")
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        pi = CaselessLiteral("PI")
        expr = Forward()
        atom = (Optional("-") + ( pi | e | fnumber | ident + lpar + expr + rpar ).setParseAction(pushFirst) | (
        lpar + expr.suppress() + rpar )).setParseAction(pushUMinus)
        factor = Forward()
        factor << atom + ZeroOrMore(( expop + factor ).setParseAction(pushFirst))
        term = factor + ZeroOrMore(( multop + factor ).setParseAction(pushFirst))
        expr << term + ZeroOrMore(( addop + term ).setParseAction(pushFirst))
        bnf = expr
    return bnf

# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12
opn = {"+": operator.add,
       "-": operator.sub,
       "*": operator.mul,
       "/": operator.truediv,
       "^": operator.pow}

math_method = [method for method in dir(math) if callable(getattr(math, method))]
fn = {"trunc": lambda a: int(a),
      "round": round,
      "e": math.exp,
      "sgn": lambda a: abs(a) > epsilon and cmp(a, 0) or 0}
for method in math_method:
    fn[method] = getattr(math, method)

def evaluateStack(s):
    op = s.pop()
    if op == 'unary -':
        return -evaluateStack(s)
    if op in "+-*/^":
        op2 = evaluateStack(s)
        op1 = evaluateStack(s)
        return opn[op](op1, op2)
    elif op == "PI":
        return math.pi  # 3.1415926535
    elif op in fn:
        return fn[op](evaluateStack(s))
    elif op[0].isalpha():
        logging.warn("Warning : Couldn't evaluate : " + op)
        return 0
    else:
        return float(op)


def fparse(expression):
    def generated(**kwargs):
        global exprStack
        exprStack = []
        fstring = expression
        for var in kwargs.keys():
            fstring = fstring.replace(var, str(kwargs[var]))
        BNF().parseString(fstring)
        val = evaluateStack(exprStack[:])
        return val
    return generated

if __name__ == "__main__":
    result = fparse("e(x)")(x=1)
    print result
