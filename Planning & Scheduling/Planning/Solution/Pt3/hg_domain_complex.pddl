; Assignment 1 - Planning & Scheduling
; Part 3 - Blackbox Planner
; Michael McAleer (R00143621)

(define (domain Forest)
  (:requirements :strips)

  (:predicates
    (at ?person ?loc)
    (adj ?loc1 ?loc2 ?dir)
  )

  (:action move_north_h
    :parameters (?from ?to)
    :precondition (and (at hansel ?from) (adj ?from ?to north))
    :effect (and (not (at hansel ?from)) (at hansel ?to))
  )

  (:action move_east_h
    :parameters (?from ?to)
    :precondition (and (at hansel ?from) (adj ?from ?to east))
    :effect (and (not (at hansel ?from)) (at hansel ?to))
  )

  (:action move_south_h
    :parameters (?from ?to)
    :precondition (and (at hansel ?from) (adj ?from ?to south))
    :effect (and (not (at hansel ?from)) (at hansel ?to))
  )

  (:action move_west_h
    :parameters (?from ?to)
    :precondition (and (at hansel ?from) (adj ?from ?to west))
    :effect (and (not (at hansel ?from)) (at hansel ?to))
  )

  (:action move_north_g
    :parameters (?from ?to)
    :precondition (and (at gretel ?from) (adj ?from ?to north))
    :effect (and (not (at gretel ?from)) (at gretel ?to))
  )

  (:action move_east_g
    :parameters (?from ?to)
    :precondition (and (at gretel ?from) (adj ?from ?to east))
    :effect (and (not (at gretel ?from)) (at gretel ?to))
  )

  (:action move_south_g
    :parameters (?from ?to)
    :precondition (and (at gretel ?from) (adj ?from ?to south))
    :effect (and (not (at gretel ?from)) (at gretel ?to))
  )

  (:action move_west_g
    :parameters (?from ?to)
    :precondition (and (at gretel ?from) (adj ?from ?to west))
    :effect (and (not (at gretel ?from)) (at gretel ?to))
  )
)
