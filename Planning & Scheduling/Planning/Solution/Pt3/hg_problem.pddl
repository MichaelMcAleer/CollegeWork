; Assignment 1 - Planning & Scheduling
; Part 3 - Blackbox Planner
; Michael McAleer (R00143621)

(define (problem Forest_Escape)

  (:domain Forest)

  (:objects
    hansel gretel
    north south west east
              c3-5 c4-5 c5-5
                   c4-4 c5-4
    c1-3 c2-3 c3-3 c5-3
    c1-2 c2-2 c3-2 c4-2 c5-2
    c1-1 c3-1      c4-1
  )

  (:init
    ; hansel and gretel locations
    (at hansel c1-2)
    (at gretel c4-4)

    ; c1-1
    (adj c1-1 c1-2 north)

    ; c1-2
    (adj c1-2 c1-1 south)
    (adj c1-2 c1-3 north)
    (adj c1-2 c2-2 east)

    ; 1-3
    (adj c1-3 c1-2 south)
    (adj c1-3 c2-3 east)

    ; 2-2
    (adj c2-2 c1-2 west)
    (adj c2-2 c2-3 north)
    (adj c2-2 c3-2 east)

    ; 2-3
    (adj c2-3 c2-2 south)
    (adj c2-3 c1-3 west)
    (adj c2-3 c3-3 east)

    ; 3-1
    (adj c3-1 c3-2 north)
    (adj c3-1 c4-1 east)

    ; 3-2
    (adj c3-2 c3-1 south)
    (adj c3-2 c2-2 west)
    (adj c3-2 c3-3 north)
    (adj c3-2 c4-2 east)

    ; 3-3
    (adj c3-3 c3-1 south)
    (adj c3-3 c2-3 west)

    ; 3-5
    (adj c3-5 c4-5 east)

    ; 4-1
    (adj c4-1 c3-1 west)
    (adj c4-1 c4-2 north)

    ; 4-2
    (adj c4-2 c4-1 south)
    (adj c4-2 c3-2 west)
    (adj c4-2 c5-2 east)

    ; 4-4
    (adj c4-4 c4-5 north)
    (adj c4-4 c5-4 west)

    ; 4-5
    (adj c4-5 c4-4 south)
    (adj c4-5 c3-5 west)
    (adj c4-5 c5-5 east)

    ; 5-2
    (adj c5-2 c4-2 west)
    (adj c5-2 c5-3 north)

    ; 5-3
    (adj c5-3 c5-4 north)
    (adj c5-3 c5-2 south)

    ; 5-4
    (adj c5-4 c5-3 south)
    (adj c5-4 c4-4 west)
    (adj c5-4 c5-5 north)

    ; 5-5
    (adj c5-5 c5-4 south)
    (adj c5-5 c4-5 west)
  )

  (:goal
    (and (at hansel c3-3)
         (at gretel c3-3)
    )
  )
)
