pub(crate) mod generator;

use crate::event::PhaseSpaceEvent;
use crate::four_vector::FourMomentum;
use generator::{boost_four_momenta, correct_masses, init_four_momenta};
use rayon::prelude::*;

/// Bare rambo struct for namespacing methods.
struct Rambo;

impl Rambo {
    /// Generate `nevents` phase-space events for a process with final state
    /// particle masses `masses`, center-of-mass energy `cme` and squared matrix
    /// element `msqrd`.
    #[allow(dead_code)]
    pub fn generate_phase_space<F>(
        masses: &Vec<f64>,
        cme: f64,
        msqrd: F,
        nevents: usize,
    ) -> Vec<PhaseSpaceEvent>
    where
        F: Fn(&Vec<FourMomentum>) -> f64 + Sync,
    {
        let n = masses.len();
        let fill = |event: &mut PhaseSpaceEvent| {
            init_four_momenta(&mut event.momenta);
            boost_four_momenta(event, cme);
            correct_masses(event, &masses, cme);
            event.weight *= msqrd(&event.momenta);
        };
        vec![PhaseSpaceEvent::new(n); nevents]
            .into_par_iter()
            .update(|ev: &mut PhaseSpaceEvent| fill(ev))
            .collect()
    }
    /// Generate `nevents` phase-space events for a process with final state
    /// particle masses `masses`, center-of-mass energy `cme` and a flat squared
    /// matrix element.
    #[allow(dead_code)]
    pub fn generate_phase_space_flat(
        masses: &Vec<f64>,
        cme: f64,
        nevents: usize,
    ) -> Vec<PhaseSpaceEvent> {
        let n = masses.len();
        let fill = |event: &mut PhaseSpaceEvent| {
            init_four_momenta(&mut event.momenta);
            boost_four_momenta(event, cme);
            correct_masses(event, &masses, cme);
        };
        vec![PhaseSpaceEvent::new(n); nevents]
            .into_par_iter()
            .update(|ev: &mut PhaseSpaceEvent| fill(ev))
            .collect()
    }
    //// Integrate over phases space for a given particle physics process with final
    /// state masses `masses`, center of mass energy `cme` and matrix element squared
    /// `msqrd` by monte-carlo integration using `nevents` number of events.
    #[allow(dead_code)]
    pub fn integrate_phase_space<F>(
        masses: &Vec<f64>,
        cme: f64,
        msqrd: F,
        nevents: usize,
    ) -> (f64, f64)
    where
        F: Fn(&Vec<FourMomentum>) -> f64 + Sync,
    {
        let events = Rambo::generate_phase_space(&masses, cme, msqrd, nevents);
        // Extract the weights in order to compute integral.
        let weights: Vec<f64> = events.iter().map(|ev| ev.weight).collect();
        // Integral is just the average of the weights.
        let avg = (&weights)
            .into_par_iter()
            .cloned()
            .reduce(|| 0.0, |acc, w| acc + w)
            / nevents as f64;
        // Compute the sum of the squared of weights needed to estimate error.
        let avg2 = weights.into_par_iter().reduce(|| 0.0, |acc, w| acc + w * w) / nevents as f64;
        // The error is the standard deviation divided by sqrt(N).
        let error = ((avg2 - avg * avg) / (nevents as f64)).abs().sqrt();
        (avg, error)
    }
    /// Integrate over phases space for a given particle physics process with final
    /// state masses `masses`, center of mass energy `cme` and a flat matrix element
    /// by monte-carlo integration using `nevents` number of events.
    #[allow(dead_code)]
    pub fn integrate_phase_space_flat(masses: &Vec<f64>, cme: f64, nevents: usize) -> (f64, f64) {
        let events = Rambo::generate_phase_space_flat(&masses, cme, nevents);
        // Extract the weights in order to compute integral.
        let weights: Vec<f64> = events.iter().map(|ev| ev.weight).collect();
        // Integral is just the average of the weights.
        let integral = (&weights)
            .into_par_iter()
            .cloned()
            .reduce(|| 0.0, |acc, w| acc + w)
            / nevents as f64;
        // Compute the sum of the squared of weights needed to estimate error.
        let sum2 = weights.into_par_iter().reduce(|| 0.0, |acc, w| acc + w * w) / nevents as f64;
        // The error is the standard deviation divided by sqrt(N).
        let error = ((sum2 - integral * integral) / (nevents as f64))
            .abs()
            .sqrt();
        (integral, error)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_integrate_muon_decay() {
        let mmuon = 105.6583745e-3;
        let melectron = 0.5109989461e-3;
        let gfermi = 1.1663787e-5;

        let msqrd = |fm: &Vec<FourMomentum>| {
            let s = (fm[1] + fm[2]).dot(&(fm[1] + fm[2]));
            let t = (fm[0] + fm[2]).dot(&(fm[0] + fm[2]));
            -16.0 * gfermi * gfermi * (s + t) * (s + t - mmuon * mmuon)
        };

        let masses = vec![melectron, 0.0, 0.0];
        let cme = mmuon;
        let (mut res, mut err) = Rambo::integrate_phase_space(&masses, cme, msqrd, 100000);
        res /= 2.0 * mmuon;
        err /= 2.0 * mmuon;
        let width = (gfermi.powi(2) * mmuon.powi(5)) / (192.0 * std::f64::consts::PI.powi(3));
        println!("result, error = ({:?}, {:?})", res, 10.0 * err);
        println!("analytic result= {:?}", width);
        assert!((width - res).abs() < err);
    }
}
