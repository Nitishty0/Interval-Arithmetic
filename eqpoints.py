import mpmath as mp

mp.mp.dps = 80

def eq_x_collinear(x, mu1, mu2):
    r1 = mp.fabs(x + mu2)
    r2 = mp.fabs(x - mu1)
    return x - mu1*(x + mu2)/r1**3 - mu2*(x - mu1)/r2**3

def solve_collinear(mu2):
    mu1 = 1 - mu2
    delta = (mu2/3)**(mp.mpf(1)/3)

    f = lambda x: eq_x_collinear(x, mu1, mu2)

    # Single good starting points — avoids secant jumping to wrong root
    xL1 = mp.findroot(f, mu1 - delta,       tol=mp.mpf('1e-70'), maxsteps=500)
    xL2 = mp.findroot(f, mu1 + delta,       tol=mp.mpf('1e-70'), maxsteps=500)
    xL3 = mp.findroot(f, -1 - mu2*5/12,    tol=mp.mpf('1e-70'), maxsteps=500)

    return xL1, xL2, xL3

def triangular_points(mu2):
    mu1 = 1 - mu2
    x = mp.mpf('0.5') - mu2   # = mu1 - 0.5
    y = mp.sqrt(3) / 2
    return x, y                # L4 and L5 share x, differ only in sign of y

def interval_validate_collinear(x_star, mu2, width=mp.mpf('1e-12')):
    """
    Validate using the full 2D effective force equations at y=0.
    Only eq1 (x-equation) is nontrivial at y=0; eq2 is identically
    zero on y=0 for any x, so we skip it.
    """
    mu1 = 1 - mu2
    xi   = mp.iv.mpf([x_star - width, x_star + width])
    mu1i = mp.iv.mpf([mu1, mu1])
    mu2i = mp.iv.mpf([mu2, mu2])

    # At y=0: r1 = |x + mu2|, r2 = |x - mu1|
    # But interval arithmetic on sqrt((x+mu2)^2) = |x+mu2| correctly
    r1 = mp.iv.sqrt((xi + mu2i)**2)
    r2 = mp.iv.sqrt((xi - mu1i)**2)

    eq1 = xi - mu1i*(xi + mu2i)/r1**3 - mu2i*(xi - mu1i)/r2**3
    return eq1.a <= 0 <= eq1.b

def interval_validate_triangular(x_star, y_star, mu2, width=mp.mpf('1e-12')):
    """Validate L4/L5 using both equations of the full 2D system."""
    mu1 = 1 - mu2
    xi   = mp.iv.mpf([x_star - width, x_star + width])
    yi   = mp.iv.mpf([y_star - width, y_star + width])
    mu1i = mp.iv.mpf([mu1, mu1])
    mu2i = mp.iv.mpf([mu2, mu2])

    r1 = mp.iv.sqrt((xi + mu2i)**2 + yi**2)
    r2 = mp.iv.sqrt((xi - mu1i)**2 + yi**2)

    eq1 = xi - mu1i*(xi + mu2i)/r1**3 - mu2i*(xi - mu1i)/r2**3
    eq2 = yi - mu1i*yi/r1**3           - mu2i*yi/r2**3

    return (eq1.a <= 0 <= eq1.b) and (eq2.a <= 0 <= eq2.b)

def compute_system(mu2):
    x1, x2, x3 = solve_collinear(mu2)
    x4, y4 = triangular_points(mu2)
    return {
        "L1": (x1, mp.mpf('0')),
        "L2": (x2, mp.mpf('0')),
        "L3": (x3, mp.mpf('0')),
        "L4": (x4,  y4),
        "L5": (x4, -y4),
    }

systems = {
    "Sun-Jupiter":      mp.mpf('9.537e-4'),
    "Sun-(Earth+Moon)": mp.mpf('3.036e-6'),
    "Earth-Moon":       mp.mpf('1.215e-2'),
    "Mars-Phobos":      mp.mpf('1.667e-8'),
    "Jupiter-Io":       mp.mpf('4.704e-5'),
    "Jupiter-Europa":   mp.mpf('2.528e-5'),
    "Jupiter-Ganymede": mp.mpf('7.804e-5'),
    "Jupiter-Callisto": mp.mpf('5.667e-5'),
    "Saturn-Mimas":     mp.mpf('6.723e-8'),
    "Saturn-Titan":     mp.mpf('2.366e-4'),
    "Neptune-Triton":   mp.mpf('2.089e-4'),
    "Pluto-Charon":     mp.mpf('1.097e-1'),
}

fmt = lambda x: mp.nstr(x, 25)

for name, mu2 in systems.items():
    pts = compute_system(mu2)
    print(f"\n{name} (mu={fmt(mu2)}):")
    for label in ["L1", "L2", "L3", "L4", "L5"]:
        x, y = pts[label]
        print(f"  {label}: x={fmt(x)},  y={fmt(y)}")

    # Interval validation
    for lab in ["L1", "L2", "L3"]:
        ok = interval_validate_collinear(pts[lab][0], mu2)
        print(f"    interval check {lab}: {'PASS' if ok else 'FAIL'}")

    for lab in ["L4", "L5"]:
        ok = interval_validate_triangular(*pts[lab], mu2)
        print(f"    interval check {lab}: {'PASS' if ok else 'FAIL'}")