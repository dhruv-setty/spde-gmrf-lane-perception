/-
  SPDEGMRF.lean — Lean 4 + Mathlib proof sketch of the structural claims in the
  README of `whatheheckisthis/SPDE-GMRF` (Section 1, "SPDE-GMRF and HMM model").

  STATUS OF EACH PART
  * §1 Barycentric projector ............ fully proved (no sorry)
  * §2 Log-determinant from Cholesky .... fully proved (no sorry)
  * §3 int8 quantisation keeps a_14 ..... fully proved (no sorry)
  * §4 Sparsity of Q (distance ≤ 2) ..... fully proved (no sorry)
  * §5 Fill-in path theorem ............. SKETCH.
      - `filled_implies_path` (symbolic elimination ⇒ path through earlier
        vertices) is fully proved.
      - `cholesky_fill_path` (numeric Cholesky entries ⇒ symbolic fill) is
        stated and left as `sorry`. This is the only `sorry` in the file.

  NOT FORMALISED: the empirical MAC counts (174,362 after RCM, 3,877,939
  scrambled) and nonzero counts are measurements, not theorems.
-/
import Mathlib

open Matrix BigOperators Finset

namespace SPDEGMRF

/-! ## §1 Barycentric projector

README: "each row of P has three barycentric weights that sum to one". -/

/-- README: "each row of P has three barycentric weights that sum to one".
If every row of `P` sums to `1`, then `P` maps a constant field to the same
constant at every sensor: constants are preserved. -/
theorem projector_preserves_constants {m n : ℕ} (P : Matrix (Fin m) (Fin n) ℝ)
    (hrow : ∀ i, ∑ j, P i j = 1) (c : ℝ) :
    P.mulVec (fun _ => c) = fun _ => c := by
  funext i
  simp only [Matrix.mulVec, dotProduct]
  rw [← Finset.sum_mul, hrow i, one_mul]

/-- README: barycentric weights are nonnegative and sum to one, so each sensor
value `(P x) i` is a convex combination of node values: it lies between any
lower bound `lo` and upper bound `hi` of `x` (in particular its min and max). -/
theorem projector_convex_bound {m n : ℕ} (P : Matrix (Fin m) (Fin n) ℝ)
    (hnn : ∀ i j, 0 ≤ P i j) (hrow : ∀ i, ∑ j, P i j = 1)
    (x : Fin n → ℝ) (lo hi : ℝ) (hlo : ∀ j, lo ≤ x j) (hhi : ∀ j, x j ≤ hi)
    (i : Fin m) : lo ≤ P.mulVec x i ∧ P.mulVec x i ≤ hi := by
  simp only [Matrix.mulVec, dotProduct]
  constructor
  · calc lo = ∑ j, P i j * lo := by rw [← Finset.sum_mul, hrow i, one_mul]
      _ ≤ ∑ j, P i j * x j :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hlo j) (hnn i j)
  · calc ∑ j, P i j * x j ≤ ∑ j, P i j * hi :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hhi j) (hnn i j)
      _ = hi := by rw [← Finset.sum_mul, hrow i, one_mul]

/-! ## §2 Log-determinant from the Cholesky factor

README: "log|Q| = 2 sum log L_ii". -/

/-- README: "one sparse Cholesky factor L gives ... log|Q|". For lower-triangular
`L`, `det (L * Lᵀ) = (∏ i, L i i)^2`. -/
theorem det_cholesky {n : ℕ} (L : Matrix (Fin n) (Fin n) ℝ)
    (hlow : ∀ i j, i < j → L i j = 0) :
    (L * Lᵀ).det = (∏ i, L i i) ^ 2 := by
  have hL : L.det = ∏ i, L i i :=
    Matrix.det_of_isLowerTriangular L (fun i j hij => hlow i j hij)
  rw [Matrix.det_mul, Matrix.det_transpose, hL, sq]

/-- README: "log|Q| = 2 sum log L_ii". With `Q = L Lᵀ`, `L` lower-triangular with
positive diagonal, `log det Q = 2 * ∑ i, log (L i i)`. -/
theorem logdet_cholesky {n : ℕ} (Q L : Matrix (Fin n) (Fin n) ℝ)
    (hQ : Q = L * Lᵀ) (hlow : ∀ i j, i < j → L i j = 0) (hpos : ∀ i, 0 < L i i) :
    Real.log Q.det = 2 * ∑ i, Real.log (L i i) := by
  rw [hQ, det_cholesky L hlow, Real.log_pow, Real.log_prod]
  · push_cast; ring
  · intro i _; exact (hpos i).ne'

/-! ## §3 int8 quantisation keeps the calm-to-panic path a_14

README: "the int8 model MUST retain a nonzero a_14 after quantisation".
The repository quantises weights as `round(a / s)` with `s = max A / 127`; we
state the rule for any scale `s > 0`, and specialise to `s = 1/127`. -/

/-- Plain symmetric int8 rounding at scale `s`. -/
noncomputable def roundQ (s a : ℝ) : ℤ := round (a / s)

/-- Proposed floor-at-one rule (the current repo quantiser has no floor):
`q(a) = max 1 (round (a / s))`. -/
noncomputable def quantPos (s a : ℝ) : ℤ := max 1 (round (a / s))

/-- The floor-at-one rule at scale `1/127`: `q(a) = max 1 (round (a * 127))`. -/
noncomputable def q127 (a : ℝ) : ℤ := max 1 (round (a * 127))

/-- README: "MUST retain a nonzero a_14". The floor-at-one quantiser never maps a
positive transition probability to zero. -/
theorem quantPos_pos (s a : ℝ) (_ha : 0 < a) : 1 ≤ quantPos s a := le_max_left _ _

/-- README: "MUST retain a nonzero a_14", for the `* 127` rule. -/
theorem q127_keeps_a14 (a : ℝ) (_ha : 0 < a) : 1 ≤ q127 a ∧ q127 a ≠ 0 := by
  have h : 1 ≤ q127 a := le_max_left _ _
  exact ⟨h, by omega⟩

/-- The floor only changes weights that plain rounding would kill: for
`a ≥ s/2` the two rules agree. -/
theorem quantPos_eq_round (s a : ℝ) (hs : 0 < s) (ha : s / 2 ≤ a) :
    quantPos s a = roundQ s a := by
  unfold quantPos roundQ
  apply max_eq_right
  have h1 : (1 : ℝ) / 2 ≤ a / s := by
    rw [le_div_iff₀ hs]; linarith
  rw [round_eq]
  have : (1 : ℝ) ≤ a / s + 1 / 2 := by linarith
  have := Int.le_floor.mpr (by exact_mod_cast this : ((1 : ℤ) : ℝ) ≤ a / s + 1 / 2)
  exact this

/-- Counterexample (why the MUST needs the floor): plain rounding sends every
`0 < a < s/2` to zero. -/
theorem roundQ_kills_small (s a : ℝ) (hs : 0 < s) (ha : 0 ≤ a) (has : a < s / 2) :
    roundQ s a = 0 := by
  unfold roundQ
  rw [round_eq, Int.floor_eq_zero_iff]
  have h0 : 0 ≤ a / s := div_nonneg ha hs.le
  have h1 : a / s < 1 / 2 := by rw [div_lt_iff₀ hs]; linarith
  constructor <;> linarith

/-- Counterexample at scale `1/127`: plain `round (a * 127)` is `0` for every
`0 < a < 1/254`, so an unfloored int8 model would drop a rare a_14. -/
theorem round127_kills_small (a : ℝ) (ha : 0 < a) (hsmall : a < 1 / 254) :
    round (a * 127) = 0 := by
  rw [round_eq, Int.floor_eq_zero_iff]
  constructor <;> nlinarith

/-- Concrete witness: `a_14 = 1/300` is positive but rounds to zero. -/
theorem exists_positive_rounds_to_zero : ∃ a : ℝ, 0 < a ∧ round (a * 127) = 0 :=
  ⟨1 / 300, by norm_num, round127_kills_small _ (by norm_num) (by norm_num)⟩

/-! ## §4 Sparsity of Q

README: "Q = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G), where C is the lumped
mass matrix and G the stiffness matrix ... a precision that is still sparse". -/

variable {n : ℕ}

/-- Mesh adjacency including the diagonal (a `SimpleGraph` is irreflexive). -/
def Near (adj : SimpleGraph (Fin n)) (i j : Fin n) : Prop := i = j ∨ adj.Adj i j

/-- Two nodes are within graph distance 2 iff they share a (reflexive) neighbour. -/
def Within2 (adj : SimpleGraph (Fin n)) (i j : Fin n) : Prop :=
  ∃ k, Near adj i k ∧ Near adj k j

/-- The SPDE precision for alpha = 2 (README formula). -/
noncomputable def Qmat (τ κ : ℝ) (C G : Matrix (Fin n) (Fin n) ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  τ ^ 2 • (κ ^ 4 • C + (2 * κ ^ 2) • G + G * C⁻¹ * G)

/-- For any diagonal `D`, `(G * D * G) i j = 0` unless `i`, `j` share a neighbour. -/
theorem GDG_sparse (adj : SimpleGraph (Fin n)) (G : Matrix (Fin n) (Fin n) ℝ)
    (hG : ∀ i j, ¬ Near adj i j → G i j = 0) (d : Fin n → ℝ) (i j : Fin n)
    (hij : ¬ Within2 adj i j) : (G * Matrix.diagonal d * G) i j = 0 := by
  rw [Matrix.mul_apply]
  apply Finset.sum_eq_zero
  intro k _
  rw [Matrix.mul_diagonal]
  by_cases hik : Near adj i k
  · have hkj : ¬ Near adj k j := fun h => hij ⟨k, hik, h⟩
    rw [hG k j hkj, mul_zero]
  · rw [hG i k hik, zero_mul, zero_mul]

/-- README: "G C^-1 G" with lumped (diagonal) `C`: the product only couples nodes
at graph distance ≤ 2. -/
theorem GCinvG_sparse (adj : SimpleGraph (Fin n)) (G : Matrix (Fin n) (Fin n) ℝ)
    (c : Fin n → ℝ) (hG : ∀ i j, ¬ Near adj i j → G i j = 0) (i j : Fin n)
    (hij : ¬ Within2 adj i j) : (G * (Matrix.diagonal c)⁻¹ * G) i j = 0 := by
  rw [Matrix.inv_diagonal]
  exact GDG_sparse adj G hG _ i j hij

/-- README: "a precision that is still sparse". With `C` the lumped (diagonal)
mass matrix and `G` supported on mesh edges, `Q i j = 0` beyond graph
distance 2. -/
theorem Q_sparse (adj : SimpleGraph (Fin n)) (τ κ : ℝ) (c : Fin n → ℝ)
    (G : Matrix (Fin n) (Fin n) ℝ) (hG : ∀ i j, ¬ Near adj i j → G i j = 0)
    (i j : Fin n) (hij : ¬ Within2 adj i j) :
    Qmat τ κ (Matrix.diagonal c) G i j = 0 := by
  have hne : i ≠ j := fun h => hij ⟨j, Or.inl h, Or.inl rfl⟩
  have hnear : ¬ Near adj i j := fun h => hij ⟨j, h, Or.inl rfl⟩
  have h3 := GCinvG_sparse adj G c hG i j hij
  simp only [Qmat, Matrix.smul_apply, Matrix.add_apply, smul_eq_mul,
    Matrix.diagonal_apply_ne _ hne, hG i j hnear, h3]
  ring

/-! ## §5 Fill-in path theorem (Rose–Tarjan 1976) — SKETCH

README: "Fill-in is controlled by ordering, so the factorisation SHOULD use a
fill-reducing ordering (reverse Cuthill-McKee, AMD or METIS nested dissection)."

Analogy with Dijkstra. Eliminating vertex `v` joins every pair of its
not-yet-eliminated neighbours, exactly as Dijkstra, on settling `v`, relaxes the
edges out of `v`. After elimination, `i — j` is an edge of the filled graph iff
`j` is *reachable* from `i` through vertices that were all "settled"
(eliminated) before both `i` and `j`. So fill is a reachability closure through
earlier-eliminated vertices. A good ordering (RCM, AMD, nested dissection)
eliminates vertices whose earlier-eliminated reachable sets stay small, so few
pairs are joined and the factor stays sparse. -/

section Fill

variable {α : Type*}

/-- A path from `i` to `j` along edges `E` whose interior vertices all satisfy `S`. -/
inductive PathVia (E : α → α → Prop) (S : α → Prop) : α → α → Prop
  | edge {i j} : E i j → PathVia E S i j
  | trans {i v j} : PathVia E S i v → S v → PathVia E S v j → PathVia E S i j

/-- Widening the allowed interior set keeps a path a path. -/
theorem PathVia.mono {E : α → α → Prop} {S T : α → Prop} (hST : ∀ v, S v → T v)
    {i j : α} (h : PathVia E S i j) : PathVia E T i j := by
  induction h with
  | edge he => exact .edge he
  | trans _ hv _ ih1 ih2 => exact .trans ih1 (hST _ hv) ih2

/-- Symbolic elimination under the order `σ` (position of each vertex):
the edges of the filled graph `G⁺`. Eliminating `v` joins its neighbours
`i`, `j` that are eliminated later (the Dijkstra "relaxation" step). -/
inductive Filled (E : α → α → Prop) (σ : α → ℕ) : α → α → Prop
  | base {i j} : E i j → Filled E σ i j
  | fill {i v j} : Filled E σ i v → Filled E σ v j → σ v < σ i → σ v < σ j →
      Filled E σ i j

/-- Rose–Tarjan, "only if" direction, symbolic form (fully proved): a filled edge
`i — j` comes from a path in the original graph whose interior vertices are all
eliminated before both `i` and `j`. -/
theorem filled_implies_path {E : α → α → Prop} {σ : α → ℕ} {i j : α}
    (h : Filled E σ i j) :
    PathVia E (fun v => σ v < min (σ i) (σ j)) i j := by
  induction h with
  | base he => exact .edge he
  | @fill i v j _ _ hvi hvj ih1 ih2 =>
    have hv : σ v < min (σ i) (σ j) := lt_min hvi hvj
    refine .trans (ih1.mono ?_) hv (ih2.mono ?_)
    · intro u hu
      exact lt_of_lt_of_le (lt_of_lt_of_le hu (min_le_right _ _)) hv.le
    · intro u hu
      exact lt_of_lt_of_le (lt_of_lt_of_le hu (min_le_left _ _)) hv.le

/-- SKETCH (uses `sorry`). Rose–Tarjan for the numeric factor: if `Q = L Lᵀ` with
`L` lower triangular after permuting by the elimination order `σ`, then a
nonzero `L i j` (`i ≠ j`) lies in the filled graph of `Q`, hence (by
`filled_implies_path`) there is a path `i ⇝ j` in the graph of `Q` through
vertices eliminated before both. The missing step is the standard induction on
the column-by-column Cholesky recurrence
`L i j = (Q i j - ∑_{k, σ k < σ j} L i k * L j k) / L j j`. -/
theorem cholesky_fill_path {n : ℕ} (Q L : Matrix (Fin n) (Fin n) ℝ)
    (σ : Fin n → ℕ) (_hσ : Function.Injective σ) (_hQ : Q = L * Lᵀ)
    (_hlow : ∀ i j, σ i < σ j → L i j = 0) (_hpos : ∀ i, 0 < L i i)
    (i j : Fin n) (hij : i ≠ j) (hL : L i j ≠ 0) :
    PathVia (fun a b => a ≠ b ∧ Q a b ≠ 0) (fun v => σ v < min (σ i) (σ j)) i j := by
  have hfill : Filled (fun a b => a ≠ b ∧ Q a b ≠ 0) σ i j := by
    sorry
  exact filled_implies_path hfill

end Fill

end SPDEGMRF
