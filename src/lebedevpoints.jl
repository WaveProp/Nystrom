"""
    lebedev_points(n)

Return the `n` point Lebedev grid on the unit sphere.
The points are obtained form the package `Lebedev`.

## Available rules

order | points
------|-------
    3 |      6
    5 |     14
    7 |     26
    9 |     38
   11 |     50
   13 |     74
   15 |     86
   17 |    110
   19 |    146
   21 |    170
   23 |    194
   25 |    230
   27 |    266
   29 |    302
   31 |    350
   35 |    434
   41 |    590
   47 |    770
   53 |    974
   59 |   1202
   65 |   1454
   71 |   1730
   77 |   2030
   83 |   2354
   89 |   2702
   95 |   3074
  101 |   3470
  107 |   3890
  113 |   4334
  119 |   4802
  125 |   5294
"""
function lebedev_points(n::Int)
    availablerules = Lebedev.rules |> values |> collect |> sort
    index = findfirst(n .<= availablerules)
    isnothing(index) && @error "Unable to return $n Lebedev points"
    n = availablerules[index]
    X,Y,Z,_ = Lebedev.lebedev_by_points(n)
    pts = [SVector{3,Float64}(x,y,z) for (x,y,z) in zip(X,Y,Z)]
    return pts
end
