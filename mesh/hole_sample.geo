// mesh parameters
lc = 1.92; lr = 0.024;
theta = Pi/24;
Point(1) = { 0.0, -15.0,  0.0, lc};
Point(2) = {15.0, -15.0,   0.0, lc};
Point(3) = {15.0,  15.0,  0.0, lc};
Point(4) = { 0.0,  15.0,  0.0, lc};

// Circle
Point(7) = {4.5, 0.0, 0.0, lr};
Point(8) = {7.5, 0.0, 0.0, lc};
Point(9) = {10.5, 0.0, 0.0,lr};
Point(5) = {7.5-3*Sin(theta), 3*Cos(theta), 0.0, lr};
Point(6) = {7.5-3*Sin(theta), -3*Cos(theta), 0.0, lr};
Point(10)= {7.5+3*Sin(theta), 3*Cos(theta), 0.0, lr};
Point(15)= {7.5+3*Sin(theta), -3*Cos(theta), 0.0, lr};

Point(11) = {7.5+3*Sin(theta), -10.2, 0.0, lr};
Point(12) = {7.5+3*Sin(theta),  10.2, 0.0, lr};
Point(13) = {7.5-3*Sin(theta),  10.2, 0.0, lr};
Point(14) = {7.5-3*Sin(theta), -10.2, 0.0, lr};

Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,1};
Circle(7)= {7,8,6};
Circle(8)= {6,8,15};
Circle(10)= {15,8,9};
Circle(11)= {9,8,10};
Circle(12)= {10,8,5};
Circle(13)= {5,8,7};

Line(14) = {12,13};
Line(15) = {14,11};
Line(16) = {11,15};
Line(17) = {10,12};
Line(18) = {13,5};
Line(19) = {6,14};

Line Loop(1) = {1,2,3,4,15,16,10,11,17,14,18,13,7,19};
Line Loop(2) = {15,16,-8,19};
Line Loop(3) = {-12,17,14,18};


Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Surface(0) = {1,2,3};

Mesh.Algorithm = 6;
//Mesh 2;
// gmsh hole_sample.geo -format msh2 -2