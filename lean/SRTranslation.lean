import Std

namespace HeytingLean.Discovery.SRTranslation

/-- Symbolic-regression variables that correspond to boundary-matrix features. -/
inductive SRVariable where
  | height_d1
  | width_d1
  | height_d2
  | width_d2
  | rank_d1
  | rank_d2
  | null_d1
  | null_d2
  deriving DecidableEq, Repr

/-- Binary operators used by the lightweight SR AST. -/
inductive SROp where
  | add
  | sub
  | mul
  | eq
  | and_
  deriving DecidableEq, Repr

/-- Symbolic-regression expression tree. -/
inductive SRExpr where
  | var : SRVariable → SRExpr
  | lit : Int → SRExpr
  | binop : SROp → SRExpr → SRExpr → SRExpr
  deriving DecidableEq, Repr

def renderVar : SRVariable → String
  | .height_d1 => "height_d1"
  | .width_d1 => "width_d1"
  | .height_d2 => "height_d2"
  | .width_d2 => "width_d2"
  | .rank_d1 => "rank_d1"
  | .rank_d2 => "rank_d2"
  | .null_d1 => "null_d1"
  | .null_d2 => "null_d2"

def renderOp : SROp → String
  | .add => "+"
  | .sub => "-"
  | .mul => "*"
  | .eq => "="
  | .and_ => "and"

def render : SRExpr → String
  | .var v => renderVar v
  | .lit n => toString n
  | .binop op lhs rhs => s!"({render lhs} {renderOp op} {render rhs})"

/-- Count feature-variable occurrences in an expression. -/
def featureArity : SRExpr → Nat
  | .var _ => 1
  | .lit _ => 0
  | .binop _ lhs rhs => featureArity lhs + featureArity rhs

end HeytingLean.Discovery.SRTranslation

