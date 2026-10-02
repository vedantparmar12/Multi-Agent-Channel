from .base_tool import BaseTool
import math
import ast
import operator

class CalculatorTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
        # Safe operators for evaluation
        self.safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Mod: operator.mod,
        }
        
        # Safe functions
        self.safe_functions = {
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
        }
    
    @property
    def name(self) -> str:
        return "calculate"
    
    @property
    def description(self) -> str:
        return "Perform mathematical calculations and evaluations"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')"
                }
            },
            "required": ["expression"]
        }
    
    def _safe_eval(self, node):
        """Safely evaluate an AST node"""
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.Name):  # Variables/constants
            if node.id in self.safe_functions:
                return self.safe_functions[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                # Expressions like 9**9**9 otherwise hang the tool
                raise ValueError("Exponent too large (limit is 1000)")
            if type(node.op) in self.safe_operators:
                return self.safe_operators[type(node.op)](left, right)
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = self._safe_eval(node.operand)
            if type(node.op) in self.safe_operators:
                return self.safe_operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        elif isinstance(node, ast.Call):  # Function calls
            func = self._safe_eval(node.func)
            args = [self._safe_eval(arg) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    def execute(self, expression: str) -> dict:
        """Execute mathematical calculation"""
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate safely
            result = self._safe_eval(tree.body)
            
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }