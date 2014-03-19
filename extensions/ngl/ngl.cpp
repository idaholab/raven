#include <stdlib.h>
#include <stdio.h>

#include "timing.h"
#include "ngl.h"


using namespace ngl;

template<typename T>
unsigned int ngl::Geometry<T>::D;

typedef float ftype;
ftype random(ftype a, ftype b) 
{
	return a  + (b - a)* (double) rand() / (double) RAND_MAX;
}

template<typename T>
void run(int d) 
{
	NGLPoint<T> p, q;
	Geometry<T>::allocate(p);
  Geometry<T>::allocate(q);
	for(int i = 0; i < d; i++) {
		p[i] = random(0,100);
		q[i] = random(0,100);
	}
	NGLPoint<T> r;
	Geometry<T>::allocate(r);
	Geometry<T>::normalize(p);
	Geometry<T>::normalize(q);
	Geometry<T>::interpolate(p,q,1,r);

	for(int k = 0; k < d; k++ ) {
		printf("%.3f     +     %3f    =  %.3f\n", p[k], q[k], r[k]);
		//printf("%02d     +     %02d    =  %02d\n", p[k], q[k], r[k]);
	}
	
	printf(" pTp = %f     qTq= %f    rTr = %f\n", Geometry<T>::dot(p,p), Geometry<T>::dot(q,q), Geometry<T>::dot(r,r));
}

template<typename T>
void test(NGLPointSet<T> &P) {
	NGLPoint<ftype> p;
	Geometry<ftype>::allocate(p);
	IndexType *neighs;
	int numNeighs;
	P.getNeighbors(p, &neighs, numNeighs);
	
}


typedef ftype* pointf;
int main(int argc, char *argv[]) 
{
  int n = argc>1? atoi(argv[1]):100;
  int d = argc>2? atoi(argv[2]): 2;
	ftype param = argc>3? atof(argv[3]):1.0;
  int kmax = argc>4? atoi(argv[4]): 100;
	Geometry<ftype>::init(d);
	
	ftype *pts = new ftype[n*d];
	for(int i = 0;i<n;i++) {
		for(int k=0;k<d;k++) {
			pts[i*d + k] = random(0,1);
		}
	}
	
	FILE *fp = fopen("points", "w");
	for(int i = 0;i<n;i++) 
	{
		for(int k=0;k<d;k++) 
		{
			fprintf(fp, "%g ", pts[i*d+k]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	
	//run<ftype>(d);
	timestamp t1 = now();

	/*
	NGLPointSet<ftype> P(pts, n);
	NGLParams<ftype> params;
	IndexType *indices;
	int numEdges;
	ngl::getGabrielGraph(P, &indices, numEdges, params);

	timestamp t2 = now();
	fprintf(stderr, "\nEllapsed time: %f s.\n", t2-t1);
	
	for(int i=0;i<numEdges;i++) 
	{
		fprintf(stdout, "%d %d\n", indices[2*i], indices[2*i+1]);
	}
	
	delete indices;
	P.destroy();
	*/
	
	EmptyRegionTest<ftype> *gg = new Gabriel<ftype>();
	gg->initialize();
	RelaxedEmptyRegionMethod<ftype> *m = new RelaxedEmptyRegionMethod<ftype>(gg);

	ANNPointSet<ftype> P(pts, n);
	NGLParams<ftype> params;
	params.iparam0 = kmax;
	P.initialize(params);

	test<ftype>(P);

	m->initialize();
	for(int k=0;k<1;k++) {
		IndexType *indices;
		int numEdges;
		m->getNeighborGraph(P, &indices, numEdges);
		
		timestamp t2 = now();
		fprintf(stderr, "\nEllapsed time: %f s.\n", t2-t1);

		for(int i=0;i<numEdges;i++) 
		{
			fprintf(stdout, "%d %d\n", indices[2*i], indices[2*i+1]);
		}

		delete indices;
	}
	gg->destroy();
	m->destroy();
	
	
	/*
	NGL<ftype>::Point p = Geometry<ftype>::allocate();
  NGL<ftype>::Point q = Geometry<ftype>::allocate();
	for(int i = 0; i < d; i++) {
		p[i] = random(0,100);
		q[i] = random(0,100);
	}
	NGL<ftype>::Point r = Geometry<ftype>::allocate();
	Geometry<ftype>::normalize(p);
	Geometry<ftype>::normalize(q);
	Geometry<ftype>::interpolate(p,q,0.5,r);
	for(int k = 0; k < d; k++ ) {
		printf("%.3f     +     %3f    =  %.3f\n", p[k], q[k], r[k]);
		//printf("%02d     +     %02d    =  %02d\n", p[k], q[k], r[k]);
	}
	printf(" pTp = %f     qTq= %f    rTr = %f\n", Geometry<ftype>::dot(p,p), Geometry<ftype>::dot(q,q), Geometry<ftype>::dot(r,r));
	 */
}

