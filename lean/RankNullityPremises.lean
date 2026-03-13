import Mathlib

namespace HeytingLean.Discovery.RankNullityPremises

abbrev rankNullityD1
    {R V0 V1 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    (D1 : V1 →ₗ[R] V0) : Prop :=
  (Module.finrank R (LinearMap.range D1) : Int) +
      (Module.finrank R (LinearMap.ker D1) : Int) =
    (Module.finrank R V1 : Int)

abbrev rankNullityD2
    {R V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D2 : V2 →ₗ[R] V1) : Prop :=
  (Module.finrank R (LinearMap.range D2) : Int) +
      (Module.finrank R (LinearMap.ker D2) : Int) =
    (Module.finrank R V2 : Int)

abbrev connectedSurface
    {R V0 V1 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    (D1 : V1 →ₗ[R] V0) : Prop :=
  (Module.finrank R (LinearMap.range D1) : Int) =
    (Module.finrank R V0 : Int) - 1

abbrev orientableSurface
    {R V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D2 : V2 →ₗ[R] V1) : Prop :=
  (Module.finrank R (LinearMap.ker D2) : Int) = 1

abbrev vanishingMiddleBetti
    {R V0 V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1) : Prop :=
  (Module.finrank R (LinearMap.ker D1) : Int) -
      (Module.finrank R (LinearMap.range D2) : Int) = 0

abbrev bettiOneValue1
    {R V0 V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1) : Prop :=
  (Module.finrank R (LinearMap.ker D1) : Int) -
      (Module.finrank R (LinearMap.range D2) : Int) = 1

abbrev bettiOneValue2
    {R V0 V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1) : Prop :=
  (Module.finrank R (LinearMap.ker D1) : Int) -
      (Module.finrank R (LinearMap.range D2) : Int) = 2

abbrev twoComponents
    {R V0 V1 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    (D1 : V1 →ₗ[R] V0) : Prop :=
  (Module.finrank R V0 : Int) - (Module.finrank R (LinearMap.range D1) : Int) = 2

theorem sphereEuler
    {R V0 V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1)
    (h1 : rankNullityD1 (R := R) (V0 := V0) (V1 := V1) D1)
    (h2 : rankNullityD2 (R := R) (V1 := V1) (V2 := V2) D2)
    (h3 : connectedSurface (R := R) (V0 := V0) (V1 := V1) D1)
    (h4 : orientableSurface (R := R) (V1 := V1) (V2 := V2) D2)
    (h5 : vanishingMiddleBetti (R := R) (V0 := V0) (V1 := V1) (V2 := V2) D1 D2) :
    (Module.finrank R V0 : Int) - (Module.finrank R V1 : Int) + (Module.finrank R V2 : Int) = 2 := by
  dsimp [rankNullityD1, rankNullityD2, connectedSurface, orientableSurface, vanishingMiddleBetti] at h1 h2 h3 h4 h5 ⊢
  omega

theorem torusEuler
    {R V0 V1 V2 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    [AddCommGroup V2] [Module R V2] [FiniteDimensional R V2]
    (D1 : V1 →ₗ[R] V0) (D2 : V2 →ₗ[R] V1)
    (h1 : rankNullityD1 (R := R) (V0 := V0) (V1 := V1) D1)
    (h2 : rankNullityD2 (R := R) (V1 := V1) (V2 := V2) D2)
    (h3 : connectedSurface (R := R) (V0 := V0) (V1 := V1) D1)
    (h4 : orientableSurface (R := R) (V1 := V1) (V2 := V2) D2)
    (h5 : bettiOneValue2 (R := R) (V0 := V0) (V1 := V1) (V2 := V2) D1 D2) :
    (Module.finrank R V0 : Int) - (Module.finrank R V1 : Int) + (Module.finrank R V2 : Int) = 0 := by
  dsimp [rankNullityD1, rankNullityD2, connectedSurface, orientableSurface, bettiOneValue2] at h1 h2 h3 h4 h5 ⊢
  omega

theorem twoComponentB0
    {R V0 V1 : Type*}
    [DivisionRing R]
    [AddCommGroup V0] [Module R V0] [FiniteDimensional R V0]
    [AddCommGroup V1] [Module R V1] [FiniteDimensional R V1]
    (D1 : V1 →ₗ[R] V0)
    (h : twoComponents (R := R) (V0 := V0) (V1 := V1) D1) :
    (Module.finrank R V0 : Int) - (Module.finrank R (LinearMap.range D1) : Int) = 2 := by
  simpa [twoComponents] using h

end HeytingLean.Discovery.RankNullityPremises

