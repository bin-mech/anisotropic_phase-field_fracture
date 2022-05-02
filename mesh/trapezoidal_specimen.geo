/*********************************************************************
 *
 *  Generate the tapered mesh
 *  
 *********************************************************************/

//mesh size
lc = 0.003; 
//geometry
L  = 1.0; 
H1 = 0.4;
H2 = 0.72;
// initial crack length and thickness
LC = 0.15; 
HC = 2.*lc; 
LC0= 0.13;
Point(1) = {0.,-0.5*H1, 0,lc};
Point(2) = {L, -0.5*H2, 0,lc};
Point(3) = {L,  0.5*H2, 0,lc};
Point(4) = {0., 0.5*H1, 0,lc};
Point(5) = {LC, 0.,  0, lc};
Point(6) = {LC0, 0.5*HC, 0., lc};
Point(7) = {LC0,-0.5*HC, 0., lc};
Point(8) = {0.,-0.5*HC,  0., lc};
Point(9) = {0., 0.5*HC,  0., lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 9};
Line(5) = {9, 6};
Line(6) = {6, 5};
Line(7) = {5, 7};
Line(8) = {7, 8};
Line(9) = {8, 1};

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
Plane Surface(1) = {1};
Geometry.LineWidth = 0.1;
Geometry.PointSize = 1.0;
Physical Surface(1) = {1};
Mesh.Algorithm = 8;
//Mesh 2;

