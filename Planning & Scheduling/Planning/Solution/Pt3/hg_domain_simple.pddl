; Assignment 1 - Planning & Scheduling
; Part 3 - Blackbox Planner
; Michael McAleer (R00143621)

(define (domain Forest)
  (:requirements :strips)

  (:predicates
    (at ?person ?loc)
    (adj ?loc1 ?loc2 ?dir)
  )

  (:action move
    :parameters (?who ?from ?to ?dir)
    :precondition (and (at ?who ?from) (adj ?from ?to ?dir))
    :effect (and (not (at ?who ?from)) (at ?who ?to))
  )
)
