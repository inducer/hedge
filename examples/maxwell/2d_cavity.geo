// a rectangular cavity with a dielectric in one region

lc = 10e-3;
height = 50e-3;
air_width = 100e-3;
dielectric_width = 50e-3;

Point(1) = {0, 0, 0, lc};
Point(2) = {air_width, 0, 0, lc};
Point(3) = {air_width+dielectric_width, 0, 0, lc};
Point(4) = {air_width+dielectric_width, height, 0, lc};
Point(5) = {air_width, height, 0, lc};
Point(6) = {0, height, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {2, 5};

Line Loop(1) = {1, 7, 5, 6};
Line Loop(2) = {2, 3, 4, -7};

Plane Surface(1) = {1};
Plane Surface(2) = {2};

Physical Surface("vacuum") = {1};
Physical Surface("dielectric") = {2};

