lc = 1.0;   // mesh size

// ------------------
// Points (CCW order)
// ------------------

Point(1) = {0, 0, 0, lc};           // bottom left corner
Point(2) = {1, 0, 0, lc};			// bottom right corner
Point(3) = {1, 0.5, 0, lc};			// middle right corner
Point(4) = {0.5, 0.5, 0, lc};	    // inside corner
Point(5) = {0.5, 1, 0, lc};		    // top middle corner
Point(6) = {0, 1, 0, lc};			// top right corner

// -----
// Lines
// -----

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,1};

// -------------------
// Line loop & surface
// -------------------

Line Loop(1) = {1,2,3,4,5,6};  // all lines forming perimeter
Plane Surface(1) = {1};
Physical Surface("Fluid") = {1};

// ------------
// Mesh control
// ------------

// forbid quads
Mesh.RecombineAll = 0;
Mesh.Recombine3DAll = 0;

// Force triangle-only meshing
Mesh.Algorithm = 5; // Delaunay = TRIANGLES ONLY

// Generate 2D mesh
Mesh 2;