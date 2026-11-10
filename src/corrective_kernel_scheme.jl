#=
This file defines all methods related to the corrective kernel estimation of ensemble averages.
=#

#=
Initialisation of common mean variables
=#

"""

    common_mean_variables(npart::Int64, neval::Int64)
    common_mean_variables(kwparams::NamedTuple)

Construct instances of [`MeanVariable`](@ref) for ``Q^\\ast = \\mathcal{V}^\\ast``, ``\\mathcal{V}^\\ast x^\\ast``,
``\\mathcal{V}^\\ast y^\\ast``, ``\\mathcal{V}^\\ast x^{\\ast ~ 2}``, ``\\mathcal{V}^\\ast x^\\ast y^\\ast``
and ``\\mathcal{V}^\\ast y^{\\ast ~ 2}``, where ``\\mathcal{V}^\\ast`` is the lumped volume of the particle,
and ``x^\\ast`` and ``y^\\ast`` its 2D coordinates.

If passed a Named Tuple, it must have a `nparticles` field for the number of particles, and fields `neval_x` and `neval_y` for
the number of rows and columns in the 2D grid.

## Arguments:
* `npart` is the number of particles location where the means of these variables are to be estimated
* `neval` is the number of nodes in the regular rectangular grid where the means of these variables are to be estimated

## Returns:
* the [`MeanVariable`](@ref)s corresponding to all 6 quantities
"""
function common_mean_variables(npart::Int64, neval::Int64)
    variable_w = MeanVariable("", npart, neval,
                              p::Particle -> p.volume)
    variable_x = MeanVariable("", npart, neval,
                              p::Particle -> p.volume*p.x)
    variable_y = MeanVariable("", npart, neval,
                              p::Particle -> p.volume*p.y,
                              p::Particle -> p.volume*p.y_left,
                              p::Particle -> p.volume*p.y_right)
    variable_xx = MeanVariable("", npart, neval,
                               p::Particle -> p.volume*p.x^2)
    variable_xy = MeanVariable("", npart, neval,
                               p::Particle -> p.volume*p.x*p.y,
                               p::Particle -> p.volume*p.x*p.y_left,
                               p::Particle -> p.volume*p.x*p.y_right)
    variable_yy = MeanVariable("", npart, neval,
                               p::Particle -> p.volume*p.y^2,
                               p::Particle -> p.volume*p.y_left^2,
                               p::Particle -> p.volume*p.y_right^2)
    return variable_w, variable_x, variable_y, variable_xx, variable_xy, variable_yy
end

common_mean_variables(kwparams::NamedTuple) = common_mean_variables(kwparams.nparticles, kwparams.neval_x * kwparams.neval_y)

#=
Corrective SPH scheme
=#

"""

    M_grad!(out::MMatrix{3, 3, Float64, 9}, average_W::Float64, gradx_W::Float64, grady_W::Float64,
            average_dX::Float64, gradx_dX::Float64, grady_dX::Float64,
            average_dY::Float64, gradx_dY::Float64, grady_dY::Float64)

Compute the matrix ``\\Gamma`` in the [^Korzelius2016] CSPM scheme and stores it in `out`
"""
function M_grad!(out::MMatrix{3, 3, Float64, 9}, average_W::Float64, gradx_W::Float64, grady_W::Float64,
                average_dX::Float64, gradx_dX::Float64, grady_dX::Float64,
                average_dY::Float64, gradx_dY::Float64, grady_dY::Float64)
    out[1,1] = average_W
    out[1,2] = average_dX
    out[1,3] = average_dY
    out[2,1] = gradx_W
    out[2,2] = gradx_dX
    out[2,3] = gradx_dY
    out[3,1] = grady_W
    out[3,2] = grady_dX
    out[3,3] = grady_dY
end

"""

    M_hess!(out::MMatrix{3, 3, Float64, 9}, hessxx_dXX::Float64, hessxy_dXX::Float64, hessyy_dXX::Float64,
            hessxx_dXY::Float64, hessxy_dXY::Float64, hessyy_dXY::Float64,
            hessxx_dYY::Float64, hessxy_dYY::Float64, hessyy_dYY::Float64)

Compute the first term of matrix ``\\mathcal{H}`` in the [^Korzelius2016] CSPM scheme and stores it in `out`
"""
function M_hess!(out::MMatrix{3, 3, Float64, 9}, hessxx_dXX::Float64, hessxy_dXX::Float64, hessyy_dXX::Float64,
                hessxx_dXY::Float64, hessxy_dXY::Float64, hessyy_dXY::Float64,
                hessxx_dYY::Float64, hessxy_dYY::Float64, hessyy_dYY::Float64)
    out[1,1] = 0.5*hessxx_dXX
    out[1,2] = 0.5*hessxx_dXY
    out[1,3] = 0.5*hessxx_dYY
    out[2,1] = 0.5*hessxy_dXX
    out[2,2] = 0.5*hessxy_dXY
    out[2,3] = 0.5*hessxy_dYY
    out[3,1] = 0.5*hessyy_dXX
    out[3,2] = 0.5*hessyy_dXY
    out[3,3] = 0.5*hessyy_dYY
end

"""

    A_hess!(out::MMatrix{3, 3, Float64, 9}, hessxx_W::Float64, hessxx_dX::Float64, hessxx_dY::Float64,
            hessxy_W::Float64, hessxy_dX::Float64, hessxy_dY::Float64,
            hessyy_W::Float64, hessyy_dX::Float64, hessyy_dY::Float64)

Compute the matrix ``\\mathcal{A}`` in the [^Korzelius2016] CSPM scheme and stores it in `out`
"""
function A_hess!(out::MMatrix{3, 3, Float64, 9}, hessxx_W::Float64, hessxx_dX::Float64, hessxx_dY::Float64,
                hessxy_W::Float64, hessxy_dX::Float64, hessxy_dY::Float64,
                hessyy_W::Float64, hessyy_dX::Float64, hessyy_dY::Float64)
    out[1,1] = hessxx_W
    out[1,2] = hessxx_dX
    out[1,3] = hessxx_dY
    out[2,1] = hessxy_W
    out[2,2] = hessxy_dX
    out[2,3] = hessxy_dY
    out[3,1] = hessyy_W
    out[3,2] = hessyy_dX
    out[3,3] = hessyy_dY
end

"""

    C_hess!(out::MMatrix{3, 3, Float64, 9}, average_dXX::Float64, average_dXY::Float64, average_dYY::Float64,
            gradx_dXX::Float64, gradx_dXY::Float64, gradx_dYY::Float64,
            grady_dXX::Float64, grady_dXY::Float64, grady_dYY::Float64)

Compute the matrix ``\\mathcal{C}`` in the [^Korzelius2016] CSPM scheme and stores it in `out`
"""
function C_hess!(out::MMatrix{3, 3, Float64, 9}, average_dXX::Float64, average_dXY::Float64, average_dYY::Float64,
                gradx_dXX::Float64, gradx_dXY::Float64, gradx_dYY::Float64,
                grady_dXX::Float64, grady_dXY::Float64, grady_dYY::Float64)
    out[1,1] = average_dXX
    out[1,2] = average_dXY
    out[1,3] = average_dYY
    out[2,1] = gradx_dXX
    out[2,2] = gradx_dXY
    out[2,3] = gradx_dYY
    out[3,1] = grady_dXX
    out[3,2] = grady_dXY
    out[3,3] = grady_dYY
end

"""

    grad_RHS!(out::MVector{3, Float64}, average::Float64, gradx::Float64, grady::Float64)

Compute the RHS of the equation defining the corrected gradient in the [^Korzelius2016] CSPM scheme and stores it in `out`
"""
function grad_RHS!(out::MVector{3, Float64}, average::Float64, gradx::Float64, grady::Float64)
    out[1] = average
    out[2] = gradx
    out[3] = grady
end

"""

    hess_RHS!(out::MVector{3, Float64}, average::Float64, gradx::Float64, grady::Float64,
              hessxx::Float64, hessxy::Float64, hessyy::Float64,
              hessxx_W::Float64, hessxx_dX::Float64, hessxx_dY::Float64,
              hessxy_W::Float64, hessxy_dX::Float64, hessxy_dY::Float64,
              hessyy_W::Float64, hessyy_dX::Float64, hessyy_dY::Float64)

Compute the RHS of the equation defining the corrected hessian in the [^Korzelius2016] CSPM scheme and stores it in `out`
"""
function hess_RHS!(out::MVector{3, Float64}, average::Float64, gradx::Float64, grady::Float64,
                  hessxx::Float64, hessxy::Float64, hessyy::Float64,
                  hessxx_W::Float64, hessxx_dX::Float64, hessxx_dY::Float64,
                  hessxy_W::Float64, hessxy_dX::Float64, hessxy_dY::Float64,
                  hessyy_W::Float64, hessyy_dX::Float64, hessyy_dY::Float64)
    out[1] = hessxx - average * hessxx_W - gradx * hessxx_dX - grady * hessxx_dY
    out[2] = hessxy - average * hessxy_W - gradx * hessxy_dX - grady * hessxy_dY
    out[3] = hessyy - average * hessyy_W - gradx * hessyy_dX - grady * hessyy_dY
end

"""

    corrective_SPH_one_variable!(xpoints::Vector{Float64}, ypoints::Vector{Float64}, variable::MeanVariable,
                                 W::MeanVariable, X::MeanVariable, Y::MeanVariable, XX::MeanVariable,
                                 XY::MeanVariable, YY::MeanVariable; is_on_grid::Bool=false)

Same as [`corrective_SPH!`](@ref), for only one [`MeanVariable`](@ref). `xpoints` and `ypoints` are the coordinates of the
points on which the mean variables were estimated by the standard kernel technique.

The [`MeanVariable`](@ref) `variable` is modified in place.
"""
function corrective_SPH_one_variable!(xpoints::Vector{Float64}, ypoints::Vector{Float64}, variable::MeanVariable,
                                      W::MeanVariable, X::MeanVariable, Y::MeanVariable, XX::MeanVariable,
                                      XY::MeanVariable, YY::MeanVariable; is_on_grid::Bool=false)

    if is_on_grid
        Q = variable.mean_eval
        Wm = W.mean_eval
        dX = @. X.mean_eval - xpoints * W.mean_eval
        dY = @. Y.mean_eval - ypoints * W.mean_eval
        dXX = @. XX.mean_eval - 2.0 * xpoints * X.mean_eval + xpoints^2 * W.mean_eval
        dXY = @. XY.mean_eval - xpoints * Y.mean_eval - ypoints * X.mean_eval + xpoints * ypoints * W.mean_eval
        dYY = @. YY.mean_eval - 2.0 * ypoints * Y.mean_eval + ypoints^2 * W.mean_eval
        Qx = variable.gradx_eval
        Wx = W.gradx_eval
        dXx = @. X.gradx_eval - xpoints * W.gradx_eval
        dYx = @. Y.gradx_eval - ypoints * W.gradx_eval
        dXXx = @. XX.gradx_eval - 2.0 * xpoints * X.gradx_eval + xpoints^2 * W.gradx_eval
        dXYx = @. XY.gradx_eval - xpoints * Y.gradx_eval - ypoints * X.gradx_eval + xpoints * ypoints * W.gradx_eval
        dYYx = @. YY.gradx_eval - 2.0 * ypoints * Y.gradx_eval + ypoints^2 * W.gradx_eval
        Qy = variable.grady_eval
        Wy = W.grady_eval
        dXy = @. X.grady_eval - xpoints * W.grady_eval
        dYy = @. Y.grady_eval - ypoints * W.grady_eval
        dXXy = @. XX.grady_eval - 2.0 * xpoints * X.grady_eval + xpoints^2 * W.grady_eval
        dXYy = @. XY.grady_eval - xpoints * Y.grady_eval - ypoints * X.grady_eval + xpoints * ypoints * W.grady_eval
        dYYy = @. YY.grady_eval - 2.0 * ypoints * Y.grady_eval + ypoints^2 * W.grady_eval
        Qxx = variable.hessxx_eval
        Wxx = W.hessxx_eval
        dXxx = @. X.hessxx_eval - xpoints * W.hessxx_eval
        dYxx = @. Y.hessxx_eval - ypoints * W.hessxx_eval
        dXXxx = @. XX.hessxx_eval - 2.0 * xpoints * X.hessxx_eval + xpoints^2 * W.hessxx_eval
        dXYxx = @. XY.hessxx_eval - xpoints * Y.hessxx_eval - ypoints * X.hessxx_eval + xpoints * ypoints * W.hessxx_eval
        dYYxx = @. YY.hessxx_eval - 2.0 * ypoints * Y.hessxx_eval + ypoints^2 * W.hessxx_eval
        Qxy = variable.hessxy_eval
        Wxy = W.hessxy_eval
        dXxy = @. X.hessxy_eval - xpoints * W.hessxy_eval
        dYxy = @. Y.hessxy_eval - ypoints * W.hessxy_eval
        dXXxy = @. XX.hessxy_eval - 2.0 * xpoints * X.hessxy_eval + xpoints^2 * W.hessxy_eval
        dXYxy = @. XY.hessxy_eval - xpoints * Y.hessxy_eval - ypoints * X.hessxy_eval + xpoints * ypoints * W.hessxy_eval
        dYYxy = @. YY.hessxy_eval - 2.0 * ypoints * Y.hessxy_eval + ypoints^2 * W.hessxy_eval
        Qyy = variable.hessyy_eval
        Wyy = W.hessyy_eval
        dXyy = @. X.hessyy_eval - xpoints * W.hessyy_eval
        dYyy = @. Y.hessyy_eval - ypoints * W.hessyy_eval
        dXXyy = @. XX.hessyy_eval - 2.0 * xpoints * X.hessyy_eval + xpoints^2 * W.hessyy_eval
        dXYyy = @. XY.hessyy_eval - xpoints * Y.hessyy_eval - ypoints * X.hessyy_eval + xpoints * ypoints*W.hessyy_eval
        dYYyy = @. YY.hessyy_eval - 2.0 * ypoints * Y.hessyy_eval + ypoints^2 * W.hessyy_eval
    else
        Q = variable.mean_part
        Wm = W.mean_part
        dX = @. X.mean_part - xpoints * W.mean_part
        dY = @. Y.mean_part - ypoints * W.mean_part
        dXX = @. XX.mean_part - 2.0 * xpoints * X.mean_part + xpoints^2 * W.mean_part
        dXY = @. XY.mean_part - xpoints * Y.mean_part - ypoints * X.mean_part + xpoints * ypoints * W.mean_part
        dYY = @. YY.mean_part - 2.0 * ypoints * Y.mean_part + ypoints^2 * W.mean_part
        Qx = variable.gradx_part
        Wx = W.gradx_part
        dXx = @. X.gradx_part - xpoints * W.gradx_part
        dYx = @. Y.gradx_part - ypoints * W.gradx_part
        dXXx = @. XX.gradx_part - 2.0 * xpoints * X.gradx_part + xpoints^2 * W.gradx_part
        dXYx = @. XY.gradx_part - xpoints * Y.gradx_part - ypoints * X.gradx_part + xpoints * ypoints * W.gradx_part
        dYYx = @. YY.gradx_part - 2.0 * ypoints * Y.gradx_part + ypoints^2 * W.gradx_part
        Qy = variable.grady_part
        Wy = W.grady_part
        dXy = @. X.grady_part - xpoints * W.grady_part
        dYy = @. Y.grady_part - ypoints * W.grady_part
        dXXy = @. XX.grady_part - 2.0 * xpoints * X.grady_part + xpoints^2 * W.grady_part
        dXYy = @. XY.grady_part - xpoints * Y.grady_part - ypoints * X.grady_part + xpoints * ypoints * W.grady_part
        dYYy = @. YY.grady_part - 2.0 * ypoints * Y.grady_part + ypoints^2 * W.grady_part
        Qxx = variable.hessxx_part
        Wxx = W.hessxx_part
        dXxx = @. X.hessxx_part - xpoints * W.hessxx_part
        dYxx = @. Y.hessxx_part - ypoints * W.hessxx_part
        dXXxx = @. XX.hessxx_part - 2.0 * xpoints * X.hessxx_part + xpoints^2 * W.hessxx_part
        dXYxx = @. XY.hessxx_part - xpoints * Y.hessxx_part - ypoints * X.hessxx_part + xpoints * ypoints * W.hessxx_part
        dYYxx = @. YY.hessxx_part - 2.0 * ypoints * Y.hessxx_part + ypoints^2 * W.hessxx_part
        Qxy = variable.hessxy_part
        Wxy = W.hessxy_part
        dXxy = @. X.hessxy_part - xpoints * W.hessxy_part
        dYxy = @. Y.hessxy_part - ypoints * W.hessxy_part
        dXXxy = @. XX.hessxy_part - 2.0 * xpoints * X.hessxy_part + xpoints^2 * W.hessxy_part
        dXYxy = @. XY.hessxy_part - xpoints * Y.hessxy_part - ypoints * X.hessxy_part + xpoints * ypoints * W.hessxy_part
        dYYxy = @. YY.hessxy_part - 2.0 * ypoints * Y.hessxy_part + ypoints^2 * W.hessxy_part
        Qyy = variable.hessyy_part
        Wyy = W.hessyy_part
        dXyy = @. X.hessyy_part - xpoints * W.hessyy_part
        dYyy = @. Y.hessyy_part - ypoints * W.hessyy_part
        dXXyy = @. XX.hessyy_part - 2.0 * xpoints * X.hessyy_part + xpoints^2 * W.hessyy_part
        dXYyy = @. XY.hessyy_part - xpoints * Y.hessyy_part - ypoints * X.hessyy_part + xpoints * ypoints * W.hessyy_part
        dYYyy = @. YY.hessyy_part - 2.0 * ypoints * Y.hessyy_part + ypoints^2 * W.hessyy_part
    end

    matrix_gradient = zeros(MMatrix{3, 3})
    RHS_gradient = zeros(MVector{3})
    solution_gradient = zeros(MVector{3})
    matrix_hessian = zeros(MMatrix{3, 3})
    A_hessian = zeros(MMatrix{3, 3})
    C_hessian = zeros(MMatrix{3, 3})
    RHS_hessian = zeros(MVector{3})
    solution_hessian = zeros(MVector{3})
    for i in eachindex(xpoints)
        M_grad!(matrix_gradient, Wm[i], Wx[i], Wy[i], dX[i], dXx[i], dXy[i], dY[i], dYx[i], dYy[i])
        grad_RHS!(RHS_gradient, Q[i], Qx[i], Qy[i])
        solution_gradient .= inv(matrix_gradient) * RHS_gradient
        Q[i], Qx[i], Qy[i] = solution_gradient[1], solution_gradient[2], solution_gradient[3]
        M_hess!(matrix_hessian, dXXxx[i], dXXxy[i], dXXyy[i], dXYxx[i], dXYxy[i], dXYyy[i], dYYxx[i], dYYxy[i], dYYyy[i])
        A_hess!(A_hessian, Wxx[i], dXxx[i], dYxx[i], Wxy[i], dXxy[i], dYxy[i], Wyy[i], dXyy[i], dYyy[i])
        C_hess!(C_hessian, dXX[i], dXY[i], dYY[i], dXXx[i], dXYx[i], dYYx[i], dXXy[i], dXYy[i], dYYy[i])
        matrix_hessian .-= 0.5 * A_hessian * inv(matrix_gradient) * C_hessian
        hess_RHS!(RHS_hessian, Q[i], Qx[i], Qy[i], Qxx[i], Qxy[i], Qyy[i], Wxx[i], dXxx[i], dYxx[i], Wxy[i], dXxy[i], dYxy[i], Wyy[i], dXyy[i], dYyy[i])
        solution_hessian .= inv(matrix_hessian) * RHS_hessian
        Qxx[i], Qxy[i], Qyy[i] = solution_hessian[1], solution_hessian[2]/2.0, solution_hessian[3]
    end

    if is_on_grid
        @. variable.mean_eval = Q
        @. variable.gradx_eval = Qx
        @. variable.grady_eval = Qy
        @. variable.hessxx_eval = Qxx
        @. variable.hessxy_eval = Qxy
        @. variable.hessyy_eval = Qyy
    else
        @. variable.mean_part = Q
        @. variable.gradx_part = Qx
        @. variable.grady_part = Qy
        @. variable.hessxx_part = Qxx
        @. variable.hessxy_part = Qxy
        @. variable.hessyy_part = Qyy
    end

end

"""

    corrective_SPH!(mean_computer::MeanComputer, W::MeanVariable, X::MeanVariable, Y::MeanVariable, XX::MeanVariable,
                    XY::MeanVariable, YY::MeanVariable, variables::MeanVariable...; is_on_grid::Bool=false)

Apply the corrective kernel estimation scheme to each [`MeanVariable`](@ref) in `variables`. The means `W`, `X`, `Y`, `XX`, `XY`
and `YY` must be computed in advance (see [Correcting the kernel-mean estimate](@ref) for explanation). The mean variables are
modified in place.

By default, the means estimated at the particle locations are corrected, based on `mean_computer.part_pos`.
If `is_on_grid == true`, the means estimated at the nodes of the regular rectangular grid are corrected instead,
based on `mean_computer.eval_pos`.
"""
function corrective_SPH!(mean_computer::MeanComputer, W::MeanVariable, X::MeanVariable, Y::MeanVariable, XX::MeanVariable,
                         XY::MeanVariable, YY::MeanVariable, variables::MeanVariable...; is_on_grid::Bool=false)
    if is_on_grid
        xpoints = mean_computer.eval_pos.x
        ypoints = mean_computer.eval_pos.y
    else
        xpoints = mean_computer.part_pos.x
        ypoints = mean_computer.part_pos.y
    end

    for variable in variables
        corrective_SPH_one_variable!(xpoints, ypoints, variable, W, X, Y, XX, XY, YY, is_on_grid=is_on_grid)
    end
end

#=
Dividing averages and derivatives by mean density
=#

"""

    divide_mean_by_density(average::Vector{Float64}, rho_mean::Vector{Float64})

Computes ``\\widetilde{Q}`` from ``\\overline{\\rho Q}`` and ``\\overline{\\rho}``
"""
function divide_mean_by_density(average::Vector{Float64}, rho_mean::Vector{Float64})
    return @. average/rho_mean
end

"""

    divide_gradient_by_density(average::Vector{Float64}, gradient::Vector{Float64},
                               rho_mean::Vector{Float64}, rho_grad::Vector{Float64})

Computes the gradient of ``\\widetilde{Q}`` from ``\\overline{\\rho Q}``, ``\\overline{\\rho}``, and their gradient
"""
function divide_gradient_by_density(average::Vector{Float64}, gradient::Vector{Float64},
                                    rho_mean::Vector{Float64}, rho_grad::Vector{Float64})
    return @. gradient/rho_mean - average*rho_grad/rho_mean^2
end

"""

    divide_hessian_by_density(average::Vector{Float64}, gradient_1::Vector{Float64},
                              gradient_2::Vector{Float64}, hessian::Vector{Float64},
                              rho_mean::Vector{Float64}, rho_grad_1::Vector{Float64},
                              rho_grad_2::Vector{Float64}, rho_hess::Vector{Float64})
                
Computes a second-order derivative of ``\\widetilde{Q}`` from ``\\overline{\\rho Q}``, ``\\overline{\\rho}``,
their gradient and their second-order derivatives.
"""
function divide_hessian_by_density(average::Vector{Float64}, gradient_1::Vector{Float64},
                                   gradient_2::Vector{Float64}, hessian::Vector{Float64},
                                   rho_mean::Vector{Float64}, rho_grad_1::Vector{Float64},
                                   rho_grad_2::Vector{Float64}, rho_hess::Vector{Float64})
    return @. hessian/rho_mean - (gradient_1*rho_grad_2 + gradient_2*rho_grad_1 + average*rho_hess)/rho_mean^2 + 2.0*average*rho_grad_1*rho_grad_2/rho_mean^3
end

"""

    divide_by_density_one_variable!(variable::MeanVariable, density::MeanVariable; is_on_grid::Bool=false)

Divide a [`MeanVariable`](@ref) by the mean `density`. The modification is done in place. The spatial derivatives are also
updated accordingly.

By default, the updated means are the ones estimated at the particle locations. If `is_on_grid == true`, the updated means
are the ones estimated at the nodes of the regular rectangular grid instead.
"""
function divide_by_density_one_variable!(variable::MeanVariable, density::MeanVariable; is_on_grid::Bool=false)
    if is_on_grid
        variable.hessxx_eval .= divide_hessian_by_density(variable.mean_eval, variable.gradx_eval, variable.gradx_eval, variable.hessxx_eval,
                                                          density.mean_eval, density.gradx_eval, density.gradx_eval, density.hessxx_eval)
        variable.hessxy_eval .= divide_hessian_by_density(variable.mean_eval, variable.gradx_eval, variable.grady_eval, variable.hessxy_eval,
                                                          density.mean_eval, density.gradx_eval, density.grady_eval, density.hessxy_eval)
        variable.hessyy_eval .= divide_hessian_by_density(variable.mean_eval, variable.grady_eval, variable.grady_eval, variable.hessyy_eval,
                                                          density.mean_eval, density.grady_eval, density.grady_eval, density.hessyy_eval)
        variable.gradx_eval .= divide_gradient_by_density(variable.mean_eval, variable.gradx_eval, density.mean_eval, density.gradx_eval)
        variable.grady_eval .= divide_gradient_by_density(variable.mean_eval, variable.grady_eval, density.mean_eval, density.grady_eval)
        variable.mean_eval .= divide_mean_by_density(variable.mean_eval, density.mean_eval)
    else
        variable.hessxx_part .= divide_hessian_by_density(variable.mean_part, variable.gradx_part, variable.gradx_part, variable.hessxx_part,
                                                          density.mean_part, density.gradx_part, density.gradx_part, density.hessxx_part)
        variable.hessxy_part .= divide_hessian_by_density(variable.mean_part, variable.gradx_part, variable.grady_part, variable.hessxy_part,
                                                          density.mean_part, density.gradx_part, density.grady_part, density.hessxy_part)
        variable.hessyy_part .= divide_hessian_by_density(variable.mean_part, variable.grady_part, variable.grady_part, variable.hessyy_part,
                                                          density.mean_part, density.grady_part, density.grady_part, density.hessyy_part)
        variable.gradx_part .= divide_gradient_by_density(variable.mean_part, variable.gradx_part, density.mean_part, density.gradx_part)
        variable.grady_part .= divide_gradient_by_density(variable.mean_part, variable.grady_part, density.mean_part, density.grady_part)
        variable.mean_part .= divide_mean_by_density(variable.mean_part, density.mean_part)
    end
end

"""

    divide_by_density!(density::MeanVariable, variables::MeanVariable...; is_on_grid::Bool=false)

Same as [`divide_by_density_one_variable!`](@ref), but for several [`MeanVariable`](@ref)s at once. Any number of arguments
can be passed to this method, and starting from the second one, they will be treated as mean variables to be divided by density.
"""
function divide_by_density!(density::MeanVariable, variables::MeanVariable...; is_on_grid::Bool=false)
    for variable in variables
        divide_by_density_one_variable!(variable, density, is_on_grid=is_on_grid)
    end
end