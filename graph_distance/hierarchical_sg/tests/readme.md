# Hierarchical SG tests

A set of simple trees as tests for hierarchical SG distance.

There is a base scene graph in `base.json` with this structure:

- kitchen
  - table
    - plate
    - apple
  - counter
    - knife
    - cutting board
      - bellpepper

There are also a number of additional scene graphs defined, each with a relatively simple difference from the base:

| test name    | change                                                                              |
| ------------ | ----------------------------------------------------------------------------------- |
| id           | id of table is changed from 2 to 200                                                |
| name         | name of table is changed from "table" to "dobby"                                    |
| bbox_good    | bbox of table is changed, but iou is still above 0.5                                |
| bbox_bad     | bbox of table is changed, and iou is less than 0.5                                  |
| move_node    | bellpepper is moved from kitchen/counter/cuttingboard to kitchen/table/plate        |
| move_subtree | cutting board (and bellpepper) is moved from kitchen/counter to kitchen/table/plate |
