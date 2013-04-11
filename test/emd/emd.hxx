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
#include <iostream>
#include <cmath>

#include <vigra/emd.hxx>

/******************************************************************************
double emd(const Signature &Signature1, const Signature &Signature2,
	  double (*Dist)(const feature_t &, const feature_t &), EMDFlow& Flow)

where

   Signature1, Signature2  signatures that their distance we want to compute.
   Dist       Pointer to the ground distance. i.e. the function that computes
              the distance between two features.
   Flow       (Optional) Instance of EMDFlow (defined in emd.h)
              where the resulting flow will be stored.

******************************************************************************/
template<typename feature_t,
    template<typename ValueType> class DistanceFunctor>
double emd(const vigra::Signature<feature_t> &Signature1,
        const vigra::Signature<feature_t> &Signature2,
	  DistanceFunctor<feature_t> const & Dist,
	  const vigra::EMDOptions& options = vigra::EMDOptions());

template<typename feature_t,
    template<typename ValueType> class DistanceFunctor>
double emd(const vigra::Signature<feature_t> &Signature1,
        const vigra::Signature<feature_t> &Signature2,
	  DistanceFunctor<feature_t> const & func,
	  vigra::EMDFlow& flow,
      const vigra::EMDOptions& options = vigra::EMDOptions());


namespace vigra {

} // namespace vigra

template<typename feature_t,
    template<typename ValueType> class DistanceFunctor>
double emd(const vigra::Signature<feature_t> &Signature1,
        const vigra::Signature<feature_t> &Signature2,
	  DistanceFunctor<feature_t> const & Dist,
	  const vigra::EMDOptions& options)
{
    return vigra::EMDComputerRubner<feature_t>(options)(Signature1, Signature2, Dist);
}


template<typename feature_t,
    template<typename ValueType> class DistanceFunctor>
double emd(const vigra::Signature<feature_t> &Signature1,
        const vigra::Signature<feature_t> &Signature2,
	  DistanceFunctor<feature_t> const & Dist,
	  vigra::EMDFlow &Flow, const vigra::EMDOptions& options)
{
    return vigra::EMDComputerRubner<feature_t>(options)(Signature1, Signature2, Dist, Flow);
}

#endif
