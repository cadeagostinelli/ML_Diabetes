         CLIPS (6.31 6/12/19)
CLIPS> (defrule high-bmi
   (bmi ?bmi)
   (test (>= ?bmi 30))
   =>
   (assert (risk-category high)))
CLIPS> (defrule high-blood-pressure
   (high-blood-pressure 1)
   =>
   (assert (risk-category high)))
CLIPS> (defrule high-cholesterol
   (high-cholesterol 1)
   =>
   (assert (risk-category high)))
CLIPS> (defrule combined-risk
   (bmi ?bmi)
   (high-blood-pressure 1)
   (test (>= ?bmi 25))
   =>
   (assert (risk-category high)))
CLIPS> (defrule moderate-bmi
   (bmi ?bmi)
   (test (and (>= ?bmi 25) (< ?bmi 30)))
   (not (risk-category high))
   =>
   (assert (risk-category moderate)))
CLIPS> (defrule moderate-blood-pressure
   (high-blood-pressure 1)
   (bmi ?bmi)
   (test (< ?bmi 30))
   (not (risk-category high))
   =>
   (assert (risk-category moderate)))
CLIPS> (defrule moderate-cholesterol
   (high-cholesterol 1)
   (bmi ?bmi)
   (test (< ?bmi 30))
   (not (risk-category high))
   =>
   (assert (risk-category moderate)))
CLIPS> (defrule low-risk
   (bmi ?bmi)
   (high-blood-pressure 0)
   (high-cholesterol 0)
   (test (< ?bmi 25))
   =>
   (assert (risk-category low)))
CLIPS> 
