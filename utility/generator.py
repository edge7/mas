l1 =\
[
    "SUPERATA_DA_POCO_ERA_SU_SHORT",
    "SUPERATA_DA_POCO_ERA_GIU_SHORT",
    "GIU_IN_SALITA_SHORT",
    "GIU_IN_DISCESA_SHORT",
    "SU_IN_SALITA_SHORT",
    "SU_IN_DISCESA_SHORT"

]

l2 = \
    [
        "SUPERATA_DA_POCO_ERA_SU_LONG",
        "SUPERATA_DA_POCO_ERA_GIU_LONG",
        "GIU_IN_SALITA_LONG",
        "GIU_IN_DISCESA_LONG",
        "SU_IN_SALITA_LONG",
        "SU_IN_DISCESA_LONG"

    ]

for primo in l1:
    for second in l2:
        print(primo + " -- " + second)