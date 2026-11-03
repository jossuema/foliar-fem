"""
LATENCY compartment for the infection model: S -> E_1 -> ... -> E_k -> I.

It gives the time axis of foliar/infection.py a measurable biological anchor.
The bare model turns inoculum into necrosis instantaneously; in black sigatoka
5 to 6 WEEKS pass between leaf infection and the necrotic spot a camera can
segment. That delay is the difference between "day 40 of the model" and "day 40
of the calendar".

WHAT THE CAMERA SEES AND WHAT THE MODEL SIMULATES

    S          healthy tissue
    E_1..E_k   tissue COLONISED BUT NOT VISIBLE (mycelium, fleck, thin streak).
               Segmentation does NOT see it. This is the missing compartment.
    I          VISIBLE necrosis: the dark spot the mask marks as a lesion. The
               only compartment observable in a photo.

That distinction is the module's contribution. Before, I was both "the infected"
and "the visible", and that is why beta_0 had to absorb the delay.

EQUATIONS (rho frozen within the step, as in infection.py)

    dS/dt   = -rho S                          rho = beta(x) Phi + alpha(x)
    dE_1/dt =  rho S      - kappa E_1
    dE_j/dt =  kappa E_{j-1} - kappa E_j       j = 2..k
    dI/dt   =  kappa E_k                       kappa = k / tau_E

    -div(D grad Phi) + Phi = Q,    Q = I + sum_j w_j E_j

Q, not I: asexual sporulation begins at Foure stage 2, LONG before the spot is
visible (18 days against 38). A model whose inoculum source were only I would
introduce a second 20-day delay that the biology does not have. The weights w_j
are NOT a free parameter: they follow from tau_asex / tau_E (see source_weights).

THE GUARANTEE THAT MUST NOT BE LOST

The generator A of the system above is a bidiagonal Metzler matrix with
zero-sum columns, i.e. the generator of a Markov chain. Hence exp(A dt) is a
STOCHASTIC matrix: non-negative entries and columns summing to 1, for ANY dt,
any rho >= 0 and any mesh. From that follow, with no clipping and no discrete
maximum principle:

    S + sum E_j + I = 1        exact to machine precision
    0 <= every compartment <= 1
    dI/dt = kappa E_k >= 0     I remains monotone

It is the same protection the exponential integrator of infection.step gives,
now for 2 + k compartments, and for the same reason: it comes from the
integrator, not from the operator. As tau_E -> 0 the map converges to the
current scheme I <- 1 - (1-I) exp(-dt rho) at order 1 in tau_E: measured on the
leaf, the severity error at 10 days is 2.4e-4 with tau_E = 0.01 d and 2.4e-5
with 0.001 d.

BIOLOGICAL FIGURES AND WHERE THEY COME FROM

Bornacelly Horta, H. (2009). "Estudio del ciclo de vida de Mycosphaerella
fijiensis en tres clones de banano (Musa AAA) en tres regiones de la Zona
Bananera del Magdalena". MSc thesis, Universidad Nacional de Colombia, Palmira.
Table 4, days from leaf emission (which the protocol takes as the moment of
infection) to each Foure stage:

    season     E1     E2     E3     E4     E5     E6
    dry       14.64  19.12  27.80  37.35  57.36  64.39
    rainy     10.54  16.26  27.20  43.96  60.18  66.90

Consuegra and Lorenzo (2004), cited in the same thesis (Sevilla, gran enano):
20, 24, 30, 34, 35, 38 days for stages 1 to 6; William clone: 20, 24, 30, 33,
36, 39. The original could NOT be accessed; it is used only to bound the range.

CTAHR PD-50 (Univ. of Hawaii, 2008): "The first symptom (chlorotic flecking)
appears about 15-20 days after infection", and the streak-to-spot evolution
takes 10-15 days more.

Hence the module's three numbers:

  tau_E    = 38 days. Infection -> SEGMENTABLE spot, which is stage 4 onwards:
             at stage 3 the streak is ~2 mm wide and at 4 px/cm does not reach a
             pixel. Verified range 34 (Consuegra) to 44 (Bornacelly, rainy). It
             is NOT a single number: it depends on climate and clone, which is
             why it is an input parameter and not a constant.
  tau_asex = 18 days. Asexual latency, infection -> conidia release = stage 2.
             Bornacelly Table 4: 19.12 (dry) / 16.26 (rainy).
  k        = 3 stages. It follows from the dispersion of the stage sequence
             itself, not from a fit: if each stage-to-stage transition were
             exponential, the sum has CV^2 = sum(mu_i^2)/(sum mu_i)^2 and the
             effective shape is k_eff = 1/CV^2. With the Table 4 increments up
             to E4 that gives 3.48 (dry) and 3.55 (rainy); with Consuegra, 2.47
             and 2.36. The central integer is 3. WARNING: that assumes
             memoryless transitions and uses MEANS across seasons, not the
             variance within a cohort, which neither source publishes. It is an
             upper bound on the dispersion, i.e. k = 3 is the CONSERVATIVE
             choice. Measured, k changes neither the grade crossing date nor the
             statistic by more than 2 pp: k = 1 suffices for everything the
             model delivers today.

What these sources do NOT give, and it has to be said: none measures the
variance between lesions of the same cohort, which is what would really fix k;
and both equate "leaf emission" with "infection", which is an approximation of
the field protocol, not a fact.

Units: centimetres and days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import infection
from .meshing import Mesh

# Threshold for switching from the recursion to the series in the integral J_m.
# With |c*dt| below this the recursion J_m = (t^m e^{-b t} - m J_{m-1})/c
# cancels; the series converges as (|c|t)^r/r!, so 24 terms give machine
# precision. Measured against mpmath at 50 digits over 1,680 combinations of
# (k, tau_E, dt, rho) including the exact confluence: maximum error 2.1e-13
# across the whole sweep and 5.6e-15 in the operating range (k <= 4, dt <= 2 d).
# For reference, scipy.linalg.expm over the same sweep is off by up to 5.9e-6
# near the confluence, so it does NOT serve as a reference there.
SERIES_CT = 0.5
SERIES_TERMS = 24


# --------------------------------------------------------------------------
# Latency parameters
# --------------------------------------------------------------------------
@dataclass
class LatencyModel:
    """
    Erlang(k, kappa) distributed delay between infection and visible necrosis.

    tau_E_day     Mean of the infection -> segmentable spot delay (Foure stage
                  4). 38.0 days; published range 34-44. See the module docstring
                  for the sources.
    k_stages      Number of stages in the chain. 3, derived from the dispersion
                  of the stage sequence itself (k_eff between 2.4 and 3.6).
                  k = 1 is the pure exponential delay (a single parameter);
                  k -> inf is a fixed delay with no dispersion.
    tau_asex_day  Asexual latency, infection -> conidia (stage 2). 18.0 days. It
                  fixes the weights w_j of the inoculum source; it is NOT a fit.
                  Setting None disables the latent source and leaves Q = I,
                  which is the conservative hypothesis (only visible tissue
                  sporulates).
    """
    tau_E_day: float = 38.0
    k_stages: int = 3
    tau_asex_day: float | None = 18.0

    def __post_init__(self):
        if self.k_stages < 1:
            raise ValueError("k_stages >= 1; with no latency use infection.step")
        if self.tau_E_day <= 0:
            raise ValueError("tau_E_day > 0")

    @property
    def kappa_per_day(self) -> float:
        """Per-stage rate. k/tau_E makes the MEAN of the delay equal tau_E."""
        return self.k_stages / self.tau_E_day

    @property
    def sd_day(self) -> float:
        """Delay standard deviation: tau_E/sqrt(k). With 38 and k=3, 21.9 days."""
        return self.tau_E_day / np.sqrt(self.k_stages)

    def source_weights(self) -> np.ndarray:
        """
        Weight w_j of each latent stage as an inoculum SOURCE, in [0, 1].

        Stage j covers the ages [(j-1) tau_E/k, j tau_E/k]. Tissue sporulates
        from tau_asex onwards, so w_j is the fraction of that window falling
        after tau_asex. With tau_E = 38, k = 3, tau_asex = 18 this gives
        [0.000, 0.579, 1.000]: the first stage contributes no inoculum, the
        second half of it, the third all of it. Zero free parameters: the three
        numbers were already fixed by the biology.
        """
        k = self.k_stages
        if self.tau_asex_day is None:
            return np.zeros(k)
        w = self.tau_E_day / k
        hi = np.arange(1, k + 1) * w
        return np.clip((hi - self.tau_asex_day) / w, 0.0, 1.0)


@dataclass
class LatentState:
    """
    State with latency. S is left IMPLICIT as 1 - E.sum(1) - I so the sum
    identity holds by construction and not by accumulation of errors.
    """
    t_day: float
    E: np.ndarray              # (n, k) latent fractions, invisible to the camera
    I: np.ndarray              # (n,)   visible necrosis: what the mask sees
    severity_pct: float        # of I, i.e. what is comparable with a photo
    latent_pct: float          # of sum E: what the photo does NOT see
    wsi: float
    psi: float

    @property
    def S(self) -> np.ndarray:
        return 1.0 - self.E.sum(axis=1) - self.I


# --------------------------------------------------------------------------
# Core: the exact exponential of the bidiagonal generator
# --------------------------------------------------------------------------
def _Jt(c: np.ndarray, t: float, b: float, m_max: int) -> list[np.ndarray]:
    """
    Jt_m = int_0^t s^m exp(-a(t-s) - b s) ds  for m = 0..m_max, with c = a - b.

    It is computed in this GROUPED form and not as e^{-at} * int s^m e^{cs} ds
    because then no intermediate factor grows: the integrand is bounded by t^m
    and there are no large exponentials cancelling each other.

    Two branches because of the REMOVABLE singularity at c = 0 (rho = kappa).
    This is not a rare laboratory case: with tau_E = 38 and k = 3,
    kappa = 0.079 /day, and the typical rho on this leaf are 0.03-0.10 /day, so
    the chain crosses c = 0 on part of the mesh in nearly every run.
    """
    c = np.asarray(c, float)
    small = np.abs(c * t) < SERIES_CT
    cs = c[small]                            # the series is evaluated only where needed
    ebt = np.exp(-b * t)
    eat = np.exp(-(c + b) * t)               # = exp(-a t)
    eat_s = eat[small]
    safe = np.where(np.abs(c) < 1e-300, 1.0, c)

    # The series terms (c t)^r / r! do NOT depend on m, so they are computed once
    # and reused for all m_max+1 orders. Without this the series is recomputed k
    # times per step and the model with latency costs 5 times the model without
    # it instead of 2.
    if cs.size:
        acc = np.empty((cs.size, SERIES_TERMS))
        acc[:, 0] = 1.0
        for r in range(1, SERIES_TERMS):
            acc[:, r] = acc[:, r - 1] * cs * t / r

    out = []
    prev = None
    for m in range(m_max + 1):
        if m == 0:
            val = (ebt - eat) / safe
        else:
            val = (t ** m * ebt - m * prev) / safe
        if cs.size:
            # stable branch: series  e^{-at} sum_r c^r t^{m+r+1} / (r! (m+r+1))
            coef = t ** (m + 1) / (m + 1.0 + np.arange(SERIES_TERMS))
            val = val.copy()
            # explicit product and not `acc @ coef`: see the note in LatentOperator.source
            val[small] = (acc * coef[None, :]).sum(axis=1) * eat_s
        out.append(val)
        prev = val
    return out


def latent_kernel(rho: np.ndarray, dt_day: float, kappa: float, k: int):
    """
    Blocks of exp(A dt) for the generator S -> E_1 -> ... -> E_k -> I.

    Returns (pS, W, P) with
        pS  (n,)      probability of still being in S
        W   (n, k)    S -> E_j
        P   (k,)      Poisson weights of the shift along the chain,
                      P[m] = e^{-kappa dt} (kappa dt)^m / m!, i.e. E_i -> E_{i+m}

    All closed form, no iteration. The three pieces are NON-NEGATIVE by
    construction: pS is an exponential, P are Poisson weights, and
    W_j = rho * kappa^{j-1}/(j-1)! * Jt_{j-1} is a product of positive things
    times the integral of a positive integrand. The mass missing to reach 1 is
    exactly what arrived at I, which is why I is computed by difference and not
    with its own formula: that way the identity S + sum E + I = 1 holds to
    machine precision and not through cancellation.
    """
    rho = np.asarray(rho, float)
    pS = np.exp(-rho * dt_day)
    Jt = _Jt(rho - kappa, dt_day, kappa, k - 1)

    W = np.empty((rho.size, k))
    fact = 1.0
    for j in range(1, k + 1):
        if j > 1:
            fact *= (j - 1)
        W[:, j - 1] = rho * kappa ** (j - 1) / fact * Jt[j - 1]

    x = kappa * dt_day
    P = np.exp(-x) * x ** np.arange(k) / np.array(
        [float(math.factorial(m)) for m in range(k)])
    return pS, W, P


def apply_kernel(E: np.ndarray, I: np.ndarray, pS, W, P):
    """
    One EXACT step of the linear system with rho frozen.

        E_new_j = sum_{i<=j} P_{j-i} E_i  +  S W_j
        I_new   = 1 - S pS - sum_j E_new_j

    I by difference: see latent_kernel. The result is the action of a stochastic
    matrix, so nothing needs clipping.
    """
    k = E.shape[1]
    S = 1.0 - E.sum(axis=1) - I
    En = np.zeros_like(E)
    for j in range(k):
        for i in range(j + 1):
            En[:, j] += P[j - i] * E[:, i]
        En[:, j] += S * W[:, j]
    In = 1.0 - S * pS - En.sum(axis=1)
    return En, In


# --------------------------------------------------------------------------
# Initial condition: the real problem of starting from a photo
# --------------------------------------------------------------------------
def stage_coefficients(lat: LatencyModel, growth_per_day: float) -> np.ndarray:
    """
    c_j such that E_j(0) = c_j * I(0) in an epidemic that had been growing as e^{rt}.

    With infection flux f(t) = f_0 e^{rt} and an Erlang(k, kappa) delay:

        E_j(0) = f_0 kappa^{j-1} / (kappa + r)^j
        I(0)   = f_0 L / r,     L = (kappa/(kappa+r))^k = E[e^{-rT}]

        => c_j = r kappa^{j-1} / ( L (kappa + r)^j )
           sum_j c_j = (1 + r/kappa)^k - 1

    Two properties that keep this from being an invented number:
      - r = 0 returns c = 0, i.e. "the epidemic was stopped".
      - kappa c_k = r exactly, so the initial flux into I is r I(0): the
        trajectory STARTS with the observed slope and there is no step at 0.
    """
    k, kap = lat.k_stages, lat.kappa_per_day
    r = float(growth_per_day)
    if r < 0:
        raise ValueError("growth_per_day >= 0: a falling severity is not sigatoka")
    if r == 0.0:
        return np.zeros(k)
    L = (kap / (kap + r)) ** k
    return np.array([r * kap ** (j - 1) / (L * (kap + r) ** j) for j in range(1, k + 1)])


def initial_latent(lop: "LatentOperator", I0: np.ndarray, growth_per_day: float,
                   spatial: str = "flux", picard: int = 25):
    """
    E(0) from I(0) and from ONE observable quantity: r, the exponential growth
    rate of severity BEFORE the photo.

    The problem is real and does not solve itself: the photo gives I(0) and says
    nothing about E(0), which is exactly the invisible tissue. Setting E(0) = 0
    amounts to claiming the leaf was infected for the first time on the day of
    the photo, which is false for any leaf that already has lesions, and it
    produces a PLATEAU: measured with the same beta_0, severity at 10 days rises
    0.007 pp (against 0.49 pp with E(0) set properly) and at 21 days reaches
    1.32 % instead of 2.50 %.

    Two different questions are kept apart:

    HOW MUCH latent tissue there is. It follows from the exponential-phase
    assumption, which is standard in plant epidemiology and is also MEASURABLE:
    r is estimated from two previous severity observations, or from the weekly
    monitoring the farm already does. The total mass is sum_j c_j times the
    visible mass (stage_coefficients).

    WHERE it is. NOT where the lesion is: a node with I = 1 is entirely necrotic
    and cannot host latent tissue. On the reference mesh, spreading E
    proportional to I saturates 1,808 nodes and leaves 62.6 % of the latent mass
    unplaced. Latent tissue sits in the HEALTHY tissue surrounding the lesions,
    which is exactly where the model itself puts its infection flux:

        spatial="flux"    E(x) proportional to rho(x) S(x), the model's flux
        spatial="lesion"  E(x) proportional to I(x), the naive version

    "flux" is a fixed point (rho depends on Q, which depends on E) and is solved
    with Picard iteration. Measured on the reference mesh it converges linearly
    with factor ~0.55 per iteration: the relative change is 3.8e-1 / 1.4e-1 /
    7.9e-2 / 1.2e-2 / 1.4e-4 at 2 / 3 / 4 / 8 / 20 iterations. The default of 25
    costs 25 transport solves, i.e. 16 ms measured, against the 413 ms of an
    84-step run: not a relevant cost.

    The spread respects the bound E_total(x) <= S(x) with level filling, so the
    total mass is conserved unless the leaf has no healthy tissue to put it in
    (reported in the deficit).

    Returns (E, info). info carries the target mass, the placed mass, the
    relative deficit and how many nodes ended up saturated.
    """
    I0 = np.asarray(I0, float)
    lat, lump = lop.lat, lop.op.lump
    k = lat.k_stages
    c = stage_coefficients(lat, growth_per_day)
    target = float(c.sum() * (lump * I0).sum())
    E = np.zeros((I0.size, k))
    if target <= 0.0:
        return E, dict(target_mass=0.0, placed_mass=0.0, deficit_rel=0.0,
                       n_saturated=0, picard_rel=0.0)

    cap = np.maximum(1.0 - I0, 0.0)
    split = c / c.sum()
    rel = 0.0
    for _ in range(max(picard, 1)):
        if spatial == "lesion":
            w = I0.copy()
        elif spatial == "flux":
            w = lop.rho(E, I0) * cap
        else:
            raise ValueError("spatial in {'flux', 'lesion'}")
        e_tot = _fill(w, cap, lump, target)
        new = e_tot[:, None] * split[None, :]
        rel = float(np.abs(new - E).sum() / max(np.abs(new).sum(), 1e-300))
        E = new
        if spatial == "lesion" or rel < 1e-6:
            break
    placed = float((lump * E.sum(axis=1)).sum())
    return E, dict(target_mass=target, placed_mass=placed,
                   deficit_rel=(target - placed) / target,
                   n_saturated=int((E.sum(axis=1) >= cap - 1e-12).sum()),
                   picard_rel=rel)


def _fill(w: np.ndarray, cap: np.ndarray, lump: np.ndarray, target: float,
          iters: int = 60) -> np.ndarray:
    """
    Spread `target` units of mass (weighted by lump) according to the weights w
    without exceeding the cap at any node. Level filling: what spills from a
    saturated node is redistributed among those that are not.
    """
    e = np.zeros_like(cap)
    rest = target
    for _ in range(iters):
        act = (w > 0) & (e < cap - 1e-15)
        s = float((lump[act] * w[act]).sum())
        if not act.any() or s <= 0 or rest <= 1e-14 * max(target, 1e-300):
            break
        e[act] = np.minimum(e[act] + rest * w[act] / s, cap[act])
        rest = target - float((lump * e).sum())
    return e


def growth_rate_from_pair(sev1_pct: float, sev2_pct: float, dt_day: float) -> float:
    """
    r = ln(sev2/sev1)/dt. It is the only new number the model with latency asks
    of the user, and it comes from the same pair of photos already needed to
    calibrate beta_0.

    MIND THE CIRCULARITY: if the SAME interval is used to fix r and to calibrate
    beta_0, the initial condition already explains 82 % of the increment and
    beta_0 is left with 18 %. Measured with gamma = +1 on the reference leaf,
    the calibrated beta_0 is 0.1388 /day with r = r_obs and 1.6729 /day with
    r = 0, i.e. a factor 12 that depends ENTIRELY on an assumption about the
    past. In a serious protocol r comes from an interval BEFORE the pair used for
    calibration, or is estimated jointly with beta_0 over three photos.
    """
    if sev1_pct <= 0 or sev2_pct <= 0 or dt_day <= 0:
        raise ValueError("positive severities and dt > 0")
    return float(np.log(sev2_pct / sev1_pct) / dt_day)


# --------------------------------------------------------------------------
# Operator and integration
# --------------------------------------------------------------------------
class LatentOperator:
    """
    Wraps an infection.InfectionOperator and adds the latent chain.

    It touches nothing in infection.py: it reuses the transport factorisation,
    beta, alpha, the WSI weights and the area. The only change is WHAT is handed
    to the transport, which is now Q = I + sum w_j E_j and not I.
    """

    def __init__(self, mesh: Mesh, frame, model: infection.InfectionModel,
                 lat: LatencyModel):
        self.op = infection.InfectionOperator(mesh, frame, model)
        self.lat = lat
        self.w = lat.source_weights()

    # ---- observables -----------------------------------------------------
    def source(self, E: np.ndarray, I: np.ndarray) -> np.ndarray:
        """
        Q: the tissue that SPORULATES. Not the same as what the camera sees.

        Explicit product and not `E @ self.w`: with Accelerate on macOS the
        matmul of an all-zero matrix raises a spurious divide-by-zero
        RuntimeWarning that pollutes the whole output.
        """
        return I + (E * self.w[None, :]).sum(axis=1)

    def rho(self, E: np.ndarray, I: np.ndarray) -> np.ndarray:
        return np.maximum(self.op.beta * self.op.transport(self.source(E, I))
                          + self.op.alpha, 0.0)

    def state(self, t_day: float, E: np.ndarray, I: np.ndarray) -> LatentState:
        sev = self.op.severity_pct(I)
        w = self.op.wsi(I)
        return LatentState(t_day=float(t_day), E=E.copy(), I=I.copy(),
                           severity_pct=sev,
                           latent_pct=self.op.severity_pct(E.sum(axis=1)),
                           wsi=w, psi=float(w / sev) if sev > 1e-12 else float("nan"))


def step_latent(lop: LatentOperator, E: np.ndarray, I: np.ndarray, dt_day: float,
                order: int = 2):
    """
    One step. Same recipe as infection.step: rho frozen, exact map, and
    predictor-corrector on rho for order 2.

    order=2 evaluates rho at the start and at the end of the step and reapplies
    the EXACT map from the original state with the average. The averaged map is
    still stochastic, so the correction does not cost the guarantee.

    Order measured in dt on the reference leaf, severity at 42 days:
        order=2   1.948 / 1.974 / 1.987 refining dt from 2 to 0.125 d
        order=1   0.957 / 0.978
    So latency does NOT degrade the order: it is still 2, same as infection.step.
    """
    kap, k = lop.lat.kappa_per_day, lop.lat.k_stages
    r1 = lop.rho(E, I)
    pS, W, P = latent_kernel(r1, dt_day, kap, k)
    E1, I1 = apply_kernel(E, I, pS, W, P)
    if order == 1:
        return E1, I1
    r2 = lop.rho(E1, I1)
    pS, W, P = latent_kernel(0.5 * (r1 + r2), dt_day, kap, k)
    return apply_kernel(E, I, pS, W, P)


def simulate_latent(mesh: Mesh, frame, model: infection.InfectionModel,
                    lat: LatencyModel, I0: np.ndarray, E0: np.ndarray,
                    horizon_day: float, dt_day: float = 0.5, order: int = 2,
                    record_every_day: float = 1.0):
    """
    Run the model with latency. Returns (LatentOperator, [LatentState]).

    Measured cost (11,015 nodes, 84 steps of 0.5 d): 83 ms with no latency,
    269 ms with k = 1 and 413 ms with k = 3. The overhead is the closed-form
    kernel, not the transport: two systems per step are still solved with the
    SAME factorisation. Still two orders of magnitude below meshing (~35 s).
    """
    lop = LatentOperator(mesh, frame, model, lat)
    I = np.clip(np.asarray(I0, float), 0.0, 1.0)
    E = np.array(E0, float, copy=True)
    states = [lop.state(0.0, E, I)]
    n_steps = int(round(horizon_day / dt_day))
    next_rec = record_every_day
    for kk in range(n_steps):
        E, I = step_latent(lop, E, I, dt_day, order=order)
        t = (kk + 1) * dt_day
        if t >= next_rec - 1e-12 or kk == n_steps - 1:
            states.append(lop.state(t, E, I))
            next_rec += record_every_day
    return lop, states


def calibrate_beta0_latent(mesh: Mesh, frame, model: infection.InfectionModel,
                           lat: LatencyModel, I0: np.ndarray, E0: np.ndarray,
                           horizon_day: float, target_severity_pct: float,
                           dt_day: float = 0.5, tol: float = 1e-4,
                           iters: int = 40):
    """
    Same as infection.calibrate_beta0 but on the model with latency.

    Returns (beta0, identifiable). identifiable is False in the TWO failure
    modes latency introduces and the latency-free model does not have:

      - from above: even with a huge beta_0 the target is not reached. If tau_E
        exceeds the horizon, tissue infected within the window does not have
        time to become visible and the observable increment is bounded.
      - from below: with beta_0 = 0 the target is already EXCEEDED, because the
        initial latent pool alone explains more than the observed increment.

    Both cases mean the same thing: the window between the two photos is short
    against tau_E and beta_0 is not identified. Returning a number without
    saying so would be inventing an infection rate. Measured on the reference
    leaf with tau_E = 38 d and a 21 d horizon, the latent pool explains 82 % of
    the increment: beta_0 rests on the remaining 18 %.
    """
    def sev(b0):
        m = infection.InfectionModel(**{**model.__dict__, "beta0_per_day": b0})
        _, ss = simulate_latent(mesh, frame, m, lat, I0, E0, horizon_day,
                                dt_day=dt_day, record_every_day=1e9)
        return ss[-1].severity_pct

    if sev(0.0) >= target_severity_pct:
        return 0.0, False
    lo, hi = 0.0, max(model.beta0_per_day, 1e-3)
    n_grow = 0
    while sev(hi) < target_severity_pct and n_grow < 30:
        hi *= 2.0
        n_grow += 1
    if sev(hi) < target_severity_pct:
        return float(hi), False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sev(mid) < target_severity_pct:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(hi, 1e-12):
            break
    return float(0.5 * (lo + hi)), True
