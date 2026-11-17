"""L block geometry (body frame) shared by lab modules; mirrors the datasheet in api.py."""
L_LEG = [(0, 0), (40, 0), (40, 110), (0, 110)]          # u in [0,40],  v in [0,110]
L_FOOT = [(40, 0), (100, 0), (100, 40), (40, 40)]       # u in [40,100], v in [0,40]
L_OUTLINE = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 110), (0, 110)]
L_GOAL = (300.0, 200.0, 1.5707963267948966)
