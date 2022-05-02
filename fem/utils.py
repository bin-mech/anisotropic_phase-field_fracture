from dolfin import *
import ufl

def inner_e(x, y, restrict_to_one_side=False, quadrature_degree=1):
    r"""The inner product of the tangential component of a vector field on all
    of the facets of the mesh (Measure objects dS and ds).

    By default, restrict_to_one_side is False. In this case, the function will
    return an integral that is restricted to both sides ('+') and ('-') of a
    shared facet between elements. You should use this in the case that you
    want to use the 'projected' version of DuranLibermanSpace.

    If restrict_to_one_side is True, then this will return an integral that is
    restricted ('+') to one side of a shared facet between elements. You should
    use this in the case that you want to use the `multipliers` version of
    DuranLibermanSpace.

    Args:
        x: DOLFIN or UFL Function of rank (2,) (vector).
        y: DOLFIN or UFL Function of rank (2,) (vector).
        restrict_to_one_side (Optional[bool]: Default is False.
        quadrature_degree (Optional[int]): Default is 1.

    Returns:
        UFL Form.
    """
    dSp = Measure('dS', metadata={'quadrature_degree': quadrature_degree})
    dsp = Measure('ds', metadata={'quadrature_degree': quadrature_degree})
    n = ufl.geometry.FacetNormal(x.ufl_domain())
    t = as_vector((-n[1], n[0]))
    a = (inner(x, t)*inner(y, t))('+')*dSp + \
        (inner(x, t)*inner(y, t))*dsp
    if not restrict_to_one_side:
        a += (inner(x, t)*inner(y, t))('-')*dSp
    return a
