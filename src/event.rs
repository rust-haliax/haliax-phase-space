use crate::four_vector::FourMomentum;

#[derive(Debug, Clone)]
pub struct PhaseSpaceEvent {
    pub momenta: Vec<FourMomentum>,
    pub weight: f64,
}

impl PhaseSpaceEvent {
    pub fn new(n: usize) -> PhaseSpaceEvent {
        PhaseSpaceEvent {
            momenta: vec![FourMomentum::new(); n],
            weight: 0.0,
        }
    }
}
