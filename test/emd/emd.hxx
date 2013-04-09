#ifndef _EMD_H
#define _EMD_H
/*
    emd.h

    Last update: 3/24/98

    An implementation of the Earth Movers Distance.
    Based of the solution for the Transportation problem as described in
    "Introduction to Mathematical Programming" by F. S. Hillier and 
    G. J. Lieberman, McGraw-Hill, 1990.

    Copyright (C) 1998 Yossi Rubner
    Computer Science Department, Stanford University
    E-Mail: rubner@cs.stanford.edu   URL: http://vision.stanford.edu/~rubner
*/

#include <limits>

#include <vigra/emd.hxx>

/* DEFINITIONS */

/*****************************************************************************/
/* feature_t SHOULD BE MODIFIED BY THE USER TO REFLECT THE FEATURE TYPE      */
typedef int feature_t;
/*****************************************************************************/


typedef struct
{
  int n;                /* Number of features in the signature */
  feature_t *Features;  /* Pointer to the features vector */
  double *Weights;       /* Pointer to the weights of the features */
} signature_t;


typedef struct
{
  int from;             /* Feature number in signature 1 */
  int to;               /* Feature number in signature 2 */
  double amount;         /* Amount of flow from "from" to "to" */
} flow_t;



double emd(signature_t *Signature1, signature_t *Signature2,
	  double (*func)(feature_t *, feature_t *),
	  flow_t *Flow, int *FlowSize,
      const vigra::EMDOptions& options = vigra::EMDOptions());


namespace vigra {

class EMDComputerRubner {
public:

    EMDComputerRubner() : options(EMDOptions())
    {
        initializeMemory();
    }

    EMDComputerRubner(const EMDOptions& options) : options(options)
    {
        initializeMemory();
    }

    double operator()(signature_t *Signature1, signature_t *Signature2,
            double (*func)(feature_t *, feature_t *),
            flow_t *Flow, int *FlowSize);

    ~EMDComputerRubner()
    {
        for (int i=0; i < options.maxSigSize + 1; ++i)
            delete [] _C[i];
        delete [] _C;
        delete [] _X;
        for (int i=0; i < options.maxSigSize + 1; ++i)
            delete [] _IsX[i];
        delete [] _IsX;
        delete [] _RowsX;
        delete [] _ColsX;
    }
protected:
    /* NEW TYPES DEFINITION */

    /* node1_t IS USED FOR SINGLE-LINKED LISTS */
    typedef struct node1_t {
        int i;
        double val;
        struct node1_t *Next;
    } node1_t;

    /* node1_t IS USED FOR DOUBLE-LINKED LISTS */
    typedef struct node2_t {
        int i, j;
        double val;
        struct node2_t *NextC;               /* NEXT COLUMN */
        struct node2_t *NextR;               /* NEXT ROW */
    } node2_t;

    /* DECLARATION OF FUNCTIONS */
    double init(signature_t *Signature1, signature_t *Signature2,
            double (*Dist)(feature_t *, feature_t *));
    void findBasicVariables(node1_t *U, node1_t *V);
    int isOptimal(node1_t *U, node1_t *V);
    int findLoop(node2_t **Loop);
    void newSol();
    void russel(double *S, double *D);
    void addBasicVariable(int minI, int minJ, double *S, double *D,
            node1_t *PrevUMinI, node1_t *PrevVMinJ,
            node1_t *UHead);
    void printSolution();

    void initializeMemory();
    int _n1, _n2;                          /* SIGNATURES SIZES */
    double **_C;                           /* THE COST MATRIX */
    node2_t *_X;                /* THE BASIC VARIABLES VECTOR */

    /* VARIABLES TO HANDLE _X EFFICIENTLY */
    node2_t *_EndX, *_EnterX;
    char **_IsX;
    node2_t **_RowsX, **_ColsX;
    double _maxW;
    double _maxC;

    const EMDOptions &options;
};

} // namespace vigra
#endif
