/************************************************************************/
/*                                                                      */
/*               Copyright 2011-2013 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include <unittest.hxx>
#include <vigra/emd.hxx>
#include <vigra/random.hxx>
#include <vigra/error.hxx>

#include "emd.hxx"

using namespace vigra;

class EarthMoverDistanceTest
{
    RandomMT19937 random;

public:
    typedef float ValueType;
    const ValueType errorTolerance;

    EarthMoverDistanceTest() : errorTolerance(1e-5)
    {
        // tests should be deterministic
        random.seed(0);
    }

    /*******************************************************************/
    // utility functions

    // Check that the flow given by Flow between signatures Signature1 and
    // Signature2 satisfies some basic properties.
    //
    // Let Signature1 and Signature2 have labels l_i and l_j with
    // corresponding weights w_i and w_j respectively.
    //
    // Consider the flow as a list of arrows each with origin i, target
    // j and weight f_ij. Then
    //
    // 1) f_ij >= 0 for all i, j
    //      the flow is only in one direction
    // 2) \sum_i f_ij <= w_i
    //      sum of arrows originating in the i-th coordinate is <= w_i
    // 3) \sum_j f_ij <= w_j
    //      sum of arrows ending in the j-th coordinate is <= w_j
    // 4) \sum_{i,j} f_ij = min(\sum_i w_i, \sum_j w_j)
    //      sum of the total flow does not exceed total available earth or
    //      space
    //
    // See page 8 of the following text for further explanation:
    //
    // http://www.cs.cmu.edu/~efros/courses/AP06/Papers/rubner-jcviu-00.pdf
    void checkFlowProperties(signature_t *Signature1,
            signature_t *Signature2, flow_t *Flow, int FlowSize)
    {
        ValueType source_total = 0;
        ValueType dest_total = 0;
        ValueType total_flow = 0;
        // Following loop
        //  - checks for positive flow,
        //  - computes total flow,
        //  - makes sure the origin and destination labels are valid
        //  - verifies that there are no duplicate arrows in the flow
        for (int i=0; i < FlowSize; ++i)
        {
            shouldMsg(Flow[i].amount >= 0, "flow in the wrong direction");
            total_flow += Flow[i].amount;

            shouldMsg(Flow[i].from >=0 && Flow[i].from < Signature1->n,
                    "Flow origin label not found in Signature");
            shouldMsg(Flow[i].to >=0 && Flow[i].to < Signature2->n,
                    "Flow destination label not found in Signature");
            for (int j=i+1; j < FlowSize; ++j)
            {
                if(Flow[i].from == Flow[j].from && Flow[i].to == Flow[j].to)
                    shouldMsg(false,
                            "flow contains duplicate arrows");
            }
        }

        // compute total initial and target weights
        for (int i=0; i < Signature1->n; ++i)
        {
            source_total += Signature1->Weights[i];
        }
        for (int i=0; i < Signature2->n; ++i)
        {
            dest_total += Signature2->Weights[i];
        }
        shouldEqualToleranceMessage(std::min(source_total, dest_total),
                total_flow, errorTolerance,
                "total flow more than minimum of total initial or target");

        // check if source bins are used within their limits
        for (int i=0; i < Signature1->n; ++i)
        {
            ValueType tot = 0;
            for (int j=0; j < FlowSize; ++j)
            {
                if(Flow[j].from == i)
                    tot += Flow[j].amount;
            }
            // FIXME: floats are notoriously error prone
            // the scaling factor below should at least be computed based
            // on precision and size of source/target signatures
            shouldMsg(tot <= Signature1->Weights[i] + errorTolerance*1e4,
                    "flow uses more earth than available in bin");
        }

        // check if target bins are not filled over their capacity
        for (int i=0; i < Signature2->n; ++i)
        {
            ValueType tot = 0;
            for (int j=0; j < FlowSize; ++j)
            {
                if(Flow[j].to == i)
                    tot += Flow[j].amount;
            }
            // FIXME: floats are notoriously error prone
            // the scaling factor below should at least be computed based
            // on precision and size of source/target signatures
            shouldMsg(tot <= Signature2->Weights[i] + errorTolerance*1e4,
                    "overflow in target bin");
        }
    }

    // Generate a random signature with given totalWeight.
    // If randomBinCount = 1, number of bins is selected randomly in
    // [1, maxBin]. Otherwise, it is taken to be maxBins.
    void generateRandomSignature(signature_t *signature,
            ValueType totalWeight, int maxBins, bool randomBinCount=1)
    {
        vigra_precondition(maxBins > 0,
                "Refusing to generate empty random signature.");
        int nBins;

        if (!randomBinCount)
            nBins = maxBins;
        else
            nBins = random.uniformInt(maxBins) + 1; // avoid 0

        signature->n = nBins;
        signature->Features = new feature_t[nBins];
        signature->Weights = new ValueType[nBins];

        ValueType currentTotal = 0;
        for (int i=0; i < nBins; ++i)
        {
            signature->Features[i] = i;
            signature->Weights[i] = random.uniform(0, totalWeight*maxBins);
            currentTotal += signature->Weights[i];
        }
        // normalize - adjust to totalWeight
        ValueType scaleFactor = currentTotal/totalWeight;
        for (int i=0; i < nBins; ++i)
        {
            signature->Weights[i] /= scaleFactor;
        }
    }

    // This is used to free signatures allocated by
    // generateRandomSignature()
    void freeSignature(signature_t *sig)
    {
        delete [] sig->Features;
        delete [] sig->Weights;
        delete sig;
    }

    /*******************************************************************/
    // distance functions
    //
    // temporary workaround distance functions which work with the
    // definition in the original emd sources.
    // Should be replaced by a functor.

    // L1 norm in one dimension
    static ValueType dist_l1(int* F1, int* F2)
    {
        return std::abs(*F1 - *F2);
    }

    // Matrix based distance as defined by the original example2
    static ValueType dist_example2(int* F1, int* F2)
    {
        // Cost matrix as defined in example2 of original sources
        // http://robotics.stanford.edu/~rubner/emd/default.htm
        static ValueType _COST[5][3] = {
            {3, 5, 2},
            {0, 2, 5},
            {1, 1, 3},
            {8, 4, 3},
            {7, 6, 5}
            };
        return _COST[*F1][*F2];
    }

    // end distance functions
    /*******************************************************************/

    // example 2 on
    // http://robotics.stanford.edu/~rubner/emd/default.htm
    void testEMD_RTG_example2()
    {
        feature_t   f1[5] = { 0, 1, 2, 3, 4 },
                    f2[3] = { 0, 1, 2 };
        ValueType   w1[5] = { 0.4, 0.2, 0.2, 0.1, 0.1 },
                    w2[3] = { 0.6, 0.2, 0.1 };
        signature_t s1 = { 5, f1, w1},
                    s2 = { 3, f2, w2};

        ValueType   e;
        flow_t      flow[7];
        int         flowSize;

        /* original source returns
         *
         * emd=1.888889
         *
         * flow:
         * from to  amount
         * 1    0   0.200000
         * 0    0   0.300000
         * 2    0   0.100000
         * 3    1   0.100000
         * 2    1   0.100000
         * 4    2   0.000000
         * 0    2   0.100000
         */
        e = emd(&s1, &s2, dist_example2, flow, &flowSize);
        shouldEqualMessage(flowSize, 7, "flow should have 7 items");
        shouldEqualToleranceMessage(e, 1.888889, errorTolerance,
                "Earth Moving Distance not within tolerance");
        checkFlowProperties(&s1, &s2, flow, flowSize);
    }

    // Test if emd() behaves properly for empty and too large source or
    // target signatures.
    void testEMDEmptyInOut()
    {
        feature_t   emptyFeatures[0] = {},
                    zeroFeatures[5] = {0, 1, 2, 3, 4},
                    nonemptyFeatures[3] = { 0, 1, 2 };
        ValueType   emptyWeights[0] = {},
                    zeroWeights[5] = {0, 0, 0, 0, 0},
                    nonemptyWeights[3] = { 1, 5, 8 };
        signature_t emptySignature = { 0, emptyFeatures, emptyWeights},
                    zeroSignature = { 5, zeroFeatures, zeroWeights},
                    nonemptySignature = { 3, nonemptyFeatures,
                        nonemptyWeights};

        ValueType   e;
        flow_t      flow[7]; // flow size is bounded by sig1->n + sig2->n - 1
        int         flowSize;

        // Empty signatures
        try
        {
        e = emd(&emptySignature, &emptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nSource signature cannot be empty!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty source signature");
        }

        try
        {
        e = emd(&nonemptySignature, &emptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTarget signature cannot be empty!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty target signature");
        }

        try
        {
        e = emd(&emptySignature, &nonemptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nSource signature cannot be empty!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty source signature");
        }

        // Too large signatures
        signature_t bigSignature = {MAX_SIG_SIZE + 1, emptyFeatures,
            emptyWeights};
        try
        {
        e = emd(&bigSignature, &nonemptySignature, dist_l1, NULL, NULL);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nemd: Signature size is limited to ");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for too big source signature");
        }
        try
        {
        e = emd(&nonemptySignature, &bigSignature, dist_l1, NULL, NULL);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nemd: Signature size is limited to ");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for too big target signature.");
        }

        // Zero filled signatures
        try
        {
        e = emd(&zeroSignature, &zeroSignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTotal weight of source signature cannot be 0!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for zero filled source signature.");
        }
        try
        {
        e = emd(&zeroSignature, &nonemptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTotal weight of source signature cannot be 0!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for zero filled source and target  signatures.");
        }
        try
        {
        e = emd(&nonemptySignature, &zeroSignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTotal weight of target signature cannot be 0!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for zero filled target signature.");
        }
    }

    // Test emd() between identical randomly generated input - output
    // signatures.
    void testEMDRandomToSelf()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        // how many random signatures should be generated for each size
        const int nTriesPerSize = 100;
        // total weight of the signature will be numberOfBins * weightFactor
        const ValueType weightFactor = 100;

        ValueType e;
        flow_t* flow;
        int flowSize;
        signature_t* sig;

        // Mighty compiler gods, make C++11 available, save us from our sins
        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        //std::vector<int> v_binBounds(binBounds,
        //        binBounds + sizeof(binBounds)/sizeof(int));
        //for(std::vector<int>::const_iterator itr = v_binBounds.begin();
        //        itr != v_binBounds.end(); ++itr)
        for (int i=0; i < lenBinBounds; ++i)
        {
            int cBound = binBounds[i];
            for (int j=0; j < nTriesPerSize; ++j)
            {
                sig = new signature_t;
                // generateRandomSignature allocates memory for sig
                generateRandomSignature(sig, cBound * weightFactor, cBound);
                // flow size is bounded by n1 + n2 - 1 where n1 and n2 is the
                // number of bins in the source and target signatures
                // respectively
                flow = new flow_t[cBound*2 - 1];
                e = emd(sig, sig, dist_l1, flow, &flowSize);
                shouldMsg(e == 0, "EMD to self should be 0");
                shouldMsg(flowSize == sig->n,
                        "Flow to self should have one arrow for each bin in the signature.");
                checkFlowProperties(sig, sig, flow, flowSize);

                // free stuff
                freeSignature(sig);
                delete [] flow;
            }
        }
    }

    // Test if emd() is symmetric  for randomly generated input pairs.
    //
    // If s1 and s2 are signatures with equal total weight, then
    // emd(s1, s2) == emd(s2, s1) since emd() is a metric.
    void testEMDRandomSymmetric()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        // how many random signatures should be generated for each size
        const int nTriesPerSize = 10;
        // total weight of the signature will be numberOfBins * weightFactor
        const ValueType weightFactor = 10;

        ValueType e1, e2;
        flow_t *oFlow, *rFlow;
        int oFlowSize, rFlowSize;
        signature_t *sSig, *dSig;

        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        for (int i=0; i < lenBinBounds; ++i)
        {
            int sBound = binBounds[i];
            for (int j=0; j < lenBinBounds; ++j)
            {
                int dBound = binBounds[j];
                // flow size is bounded by sBound + dBound - 1
                oFlow = new flow_t[sBound + dBound  - 1];
                rFlow = new flow_t[sBound + dBound  - 1];
                for (int k=0; k < nTriesPerSize; ++k)
                {
                    sSig = new signature_t;
                    dSig = new signature_t;
                    // generateRandomSignature allocates memory for sig
                    generateRandomSignature(sSig, sBound*dBound * weightFactor, sBound);
                    generateRandomSignature(dSig, sBound*dBound * weightFactor, dBound);
                    e1 = emd(sSig, dSig, dist_l1, oFlow, &oFlowSize);
                    checkFlowProperties(sSig, dSig, oFlow, oFlowSize);

                    e2 = emd(dSig, sSig, dist_l1, rFlow, &rFlowSize);
                    checkFlowProperties(dSig, sSig, rFlow, rFlowSize);
                    shouldEqualToleranceMessage(e1, e2, errorTolerance,
                            "emd(s1, s2) != emd(s2, s1)");
                    shouldMsg(oFlowSize == rFlowSize,
                            "Flows from emd(s1, s2) and emd(s2, s1) should have the same size.");

                    // free stuff
                    freeSignature(sSig);
                    freeSignature(dSig);
                }
                delete [] oFlow;
                delete [] rFlow;
            }
        }
    }

    // Test output of emd() on source and target signatures containing only
    // one entry.
    void testEMDRandomOneToOne()
    {
        const int nTries = 100;
        const ValueType maxWeight = 1000;
        const int maxBinIndex = 1000;

        feature_t   sourceFeatures[1] = {0},
                    targetFeatures[1] = {0};
        ValueType   sourceWeights[1] = {0},
                    targetWeights[1] = {0};
        signature_t sourceSignature = {1, sourceFeatures, sourceWeights},
                    targetSignature = { 1, targetFeatures, targetWeights};

        ValueType   e;
        flow_t      flow[1]; // flow size is bounded by sig1->n + sig2->n - 1
        int         flowSize;
        int sBinIndex, dBinIndex;
        ValueType weight;

        for (int i=0; i < nTries; ++i)
        {
            sBinIndex = random.uniformInt(maxWeight);
            dBinIndex = random.uniformInt(maxWeight);
            weight = random.uniform(0., maxWeight);

            sourceSignature.Features[0] = sBinIndex;
            targetSignature.Features[0] = dBinIndex;
            sourceSignature.Weights[0] = weight;
            targetSignature.Weights[0] = weight;

            e = emd(&sourceSignature, &targetSignature,
                    dist_l1, flow, &flowSize);
            shouldMsg(flowSize == 1,
                    "Flow between single entry signatures should have 1 element.");
            shouldEqualToleranceMessage(e, dist_l1(&sBinIndex, &dBinIndex),
                    errorTolerance, "EMD between single entry signatures not in expected tolerance.");
        }
    }

    // Test output of emd() with source or target signature with single entry.
    void testEMDRandomOneMany()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        const int nTriesPerSize = 100;
        const ValueType maxWeight = 1000;
        const int maxBinIndex = 1000;

        feature_t   singleFeatures[1] = {0};
        ValueType   singleWeights[1] = {0};
        signature_t singleSignature = {1, singleFeatures, singleWeights};

        signature_t *manySig;

        ValueType   e;
        flow_t      *flow;
        int         flowSize;
        int sBinIndex, dBinIndex;
        ValueType weight;
        ValueType expected;

        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        for (int i=0; i < lenBinBounds; ++i)
        {
            int cBound = binBounds[i];
            for (int j=0; j < nTriesPerSize; ++j)
            {
                sBinIndex = random.uniformInt(maxWeight);
                weight = random.uniform(0., maxWeight);
                singleSignature.Features[0] = sBinIndex;
                singleSignature.Weights[0] = weight;

                manySig = new signature_t;
                generateRandomSignature(manySig, weight, cBound);
                flow = new flow_t[cBound];

                e = emd(&singleSignature, manySig, dist_l1, flow, &flowSize);
                shouldMsg(flowSize == manySig->n,
                        "Flow should have one arrow for each bin in the target signature.");
                checkFlowProperties(&singleSignature, manySig, flow,
                        flowSize);

                // calculate expected value
                expected = 0;
                for(int k=0; k < manySig->n; ++k)
                {
                    expected += dist_l1(manySig->Features + k,
                            singleSignature.Features) * manySig->Weights[k];
                }
                expected /= weight;

                shouldEqualToleranceMessage(e, expected, errorTolerance,
                        "emd not equal to expected value.");

                e = emd(manySig, &singleSignature, dist_l1, flow, &flowSize);
                shouldMsg(flowSize == manySig->n,
                        "Flow should have one arrow for each bin in the source signature.");
                checkFlowProperties(manySig, &singleSignature, flow,
                        flowSize);
                shouldEqualToleranceMessage(e, expected, errorTolerance,
                        "emd not equal to expected value.");

                freeSignature(manySig);
                delete [] flow;
            }
        }
    }
};



struct HistogramDistanceTestSuite : public vigra::test_suite
{
    HistogramDistanceTestSuite()
        : vigra::test_suite("HistogramDistanceTestSuite")
    {
        add(testCase(&EarthMoverDistanceTest::testEMD_RTG_example2));
        add(testCase(&EarthMoverDistanceTest::testEMDEmptyInOut));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomToSelf));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomSymmetric));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomOneToOne));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomOneMany));
    }
};


int main (int argc, char ** argv)
{
    HistogramDistanceTestSuite test;
    const int failed = test.run(vigra::testsToBeExecuted(argc, argv));
    std::cout << test.report() << std::endl;

    return failed != 0;
}
