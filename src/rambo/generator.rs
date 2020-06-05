//! Submodule for generating phase-space events using the RAMBO algorithm.
use crate::event::PhaseSpaceEvent;
use crate::four_vector::FourMomentum;
use rand::prelude::*;
use rayon::prelude::*;

/// Generate `n` isotropic, random four-vectors with energies, q0, distributed
/// according to q0 * exp(-q0).
#[allow(dead_code)]
pub(super) fn init_four_momenta(momenta: &mut Vec<FourMomentum>) {
    let mut rng = thread_rng();

    for fm in momenta.iter_mut() {
        let rho1: f64 = rng.gen();
        let rho2: f64 = rng.gen();
        let rho3: f64 = rng.gen();
        let rho4: f64 = rng.gen();

        let c = 2.0 * rho1 - 1.0;
        let hyp = (1.0 - c * c).sqrt();
        let phi = 2.0 * rho2 * std::f64::consts::PI;
        let qe = -(rho3 * rho4).ln();

        fm.e = qe;
        fm.px = qe * hyp * phi.cos();
        fm.py = qe * hyp * phi.sin();
        fm.pz = qe * c;
    }
}

/// Boost four-momenta into the center-of-mass frame and compute the weight of
/// the event.
#[allow(dead_code)]
pub(super) fn boost_four_momenta(event: &mut PhaseSpaceEvent, cme: f64) {
    // Sum all the four-momenta
    let sum_fm = event
        .momenta
        .iter()
        .fold(FourMomentum::new(), |acc, fm| acc + *fm);
    let mass = sum_fm.mass();

    let bx = -sum_fm.px / mass;
    let by = -sum_fm.py / mass;
    let bz = -sum_fm.pz / mass;
    let x = cme / mass;
    let gamma = sum_fm.e / mass;
    let a = 1.0 / (1.0 + gamma);

    for fm in event.momenta.iter_mut() {
        let qe = fm.e;
        let qx = fm.px;
        let qy = fm.py;
        let qz = fm.pz;

        let b_dot_q = bx * qx + by * qy + bz * qz;
        fm.e = x * (gamma * qe + b_dot_q);
        fm.px = x * (qx + bx * qe + a * b_dot_q * bx);
        fm.py = x * (qy + by * qe + a * b_dot_q * by);
        fm.pz = x * (qz + bz * qe + a * b_dot_q * bz);
    }

    let pi = std::f64::consts::PI;
    let n = event.momenta.len() as i32;
    // Comute (n-2)! and (n-1)!
    let fact = if n > 3 {
        (2..(n - 1)).rev().fold(1, |acc, x| x * acc)
    } else {
        1
    };
    let fact2 = (n - 1) * fact;
    // Compute and return the weight of the event
    event.weight = (pi / 2.0).powi(n - 1) * cme.powi(2 * n - 4) * (2.0 * pi).powi(4 - 3 * n)
        / ((fact * fact2) as f64);
}

/// Compute the scale factor needed to transform a massless four-momentum in
/// a massive four-momentum with correct mass.
fn compute_scale_factor(fms: &Vec<FourMomentum>, masses: &Vec<f64>, cme: f64) -> f64 {
    let max_iter = 50;
    let tol = 1e-4;
    let mass_sum: f64 = masses.iter().sum();
    let mut xi = (1.0 - (mass_sum / cme) * (mass_sum / cme)).sqrt();

    let mut iter_count = 0;
    loop {
        let mut f = -cme;
        let mut df = 0.0;

        for (m, fm) in masses.iter().zip(fms.iter()) {
            let e2 = fm.e * fm.e;
            let del = (m * m + xi * xi * e2).sqrt();
            f += del;
            df += xi * e2 / del;
        }

        // Newton correction
        let dxi = -f / df;
        xi += dxi;
        iter_count += 1;
        if dxi.abs() < tol || iter_count >= max_iter {
            break;
        }
    }
    xi
}

#[allow(dead_code)]
pub(super) fn correct_masses(event: &mut PhaseSpaceEvent, masses: &Vec<f64>, cme: f64) {
    let mut term1 = 0.0;
    let mut term2 = 0.0;
    let mut term3 = 1.0;

    let xi = compute_scale_factor(&event.momenta, masses, cme);

    for (m, fm) in masses.iter().zip(event.momenta.iter_mut()) {
        fm.e = (m * m + xi * xi * fm.e * fm.e).sqrt();
        fm.px *= xi;
        fm.py *= xi;
        fm.pz *= xi;

        let modulus = (fm.px * fm.px + fm.py * fm.py + fm.pz * fm.pz).sqrt();

        term1 += modulus / cme;
        term2 += modulus * modulus / fm.e;
        term3 *= modulus / fm.e;
    }
    let n = masses.len() as i32;
    term1 = term1.powi(2 * n - 3);
    term2 = term2.recip();

    event.weight *= term1 * term2 * term3 * cme;
}
