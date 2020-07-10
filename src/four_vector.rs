#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourMomentum {
    pub e: f64,
    pub px: f64,
    pub py: f64,
    pub pz: f64,
}

impl FourMomentum {
    /// Create a new four-momentum with zeros for all components.
    pub fn new() -> FourMomentum {
        FourMomentum {
            e: 0.0,
            px: 0.0,
            py: 0.0,
            pz: 0.0,
        }
    }
    /// Compute the squared mass of a four-momentum.
    pub fn squared_mass(&self) -> f64 {
        self.e * self.e - self.px * self.px - self.py * self.py - self.pz * self.pz
    }
    /// Compute the mass of a four-momentum. If the squared mass is negative,
    /// then the return value is : -sqrt(-m^2).
    pub fn mass(&self) -> f64 {
        let m2 = self.squared_mass();

        if m2 < 0.0 {
            -(-m2).sqrt()
        } else {
            m2.sqrt()
        }
    }
    pub fn dot(&self, fm: &FourMomentum) -> f64 {
        self.e * fm.e - self.px * fm.px - self.py * fm.py - self.pz * fm.pz
    }
}

impl std::ops::Add for FourMomentum {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            e: self.e + other.e,
            px: self.px + other.px,
            py: self.py + other.py,
            pz: self.pz + other.pz,
        }
    }
}
impl std::ops::AddAssign for FourMomentum {
    fn add_assign(&mut self, other: Self) {
        self.e += other.e;
        self.px += other.px;
        self.py += other.py;
        self.pz += other.pz;
    }
}

impl std::ops::Sub for FourMomentum {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            e: self.e - other.e,
            px: self.px - other.px,
            py: self.py - other.py,
            pz: self.pz - other.pz,
        }
    }
}
impl std::ops::SubAssign for FourMomentum {
    fn sub_assign(&mut self, other: Self) {
        self.e -= other.e;
        self.px -= other.px;
        self.py -= other.py;
        self.pz -= other.pz;
    }
}
impl std::ops::Neg for FourMomentum {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            e: -self.e,
            px: -self.px,
            py: -self.py,
            pz: -self.pz,
        }
    }
}
