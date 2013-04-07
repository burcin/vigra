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

#include "emd.hxx"

using namespace vigra;

class EarthMoverDistanceTest
{
public:

    EarthMoverDistanceTest()
    {}

    typedef float ValueType;

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
        // check for positive flow, compute total flow and make sure the
        // origin and destination labels on the flow correspond to valid
        // labels of the source and target signatures
        for (int i=0; i < FlowSize; ++i)
        {
            shouldMsg(Flow[i].amount >= 0, "flow in the wrong direction");
            total_flow += Flow[i].amount;

            shouldMsg(Flow[i].from >=0 && Flow[i].from < Signature1->n,
                    "Flow origin label not found in Signature");
            shouldMsg(Flow[i].to >=0 && Flow[i].to < Signature2->n,
                    "Flow destination label not found in Signature");
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
        shouldMsg(total_flow == std::min(source_total, dest_total),
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
            shouldMsg(tot <= Signature1->Weights[i],
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
            shouldMsg(tot <= Signature2->Weights[i],
                    "overflow in target bin");
        }
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
        shouldEqualToleranceMessage(e, 1.888889, 1e-6,
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
};



struct HistogramDistanceTestSuite : public vigra::test_suite
{
    HistogramDistanceTestSuite()
        : vigra::test_suite("HistogramDistanceTestSuite")
    {
        add(testCase(&EarthMoverDistanceTest::testEMD_RTG_example2));
        add(testCase(&EarthMoverDistanceTest::testEMDEmptyInOut));
    }
};


int main (int argc, char ** argv)
{
    HistogramDistanceTestSuite test;
    const int failed = test.run(vigra::testsToBeExecuted(argc, argv));
    std::cout << test.report() << std::endl;

    return failed != 0;
}
