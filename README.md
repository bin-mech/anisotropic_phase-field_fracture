# Anisotropic_phase-field_fracture
FEniCS implementation of a hihger-order phase-field fracture model for brittle anisotropic materials. 
The phase-field model includes the anisotropy in surface energy as well as in elastic strain energy.
To address the possible crack surfaces interpenetration issue in compressive crack,
a recent new formulation of the orthogonal decomposition of the strain tensor, well founded 
in both the isotropic and anisotropic cases, is employed. 
The Mixed Finite Element formulation is employed to circumvent the C1 continuity for the interpolation of the phase-field parameter when using a continuous Galerkin method. This implementation based on [Fenics-Shells](https://bitbucket.org/unilucompmech/fenics-shells/src/master/) and [FEniCS Project](http://fenicsproject.org).