#define PY_ARRAY_UNIQUE_SYMBOL vigranumpylearning_PyArray_API
#define NO_IMPORT_ARRAY

#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/random_forest/hough_forest.hxx>

namespace python = boost::python;

namespace vigra
{

#ifdef HasHDF5
template<class LabelType>
Hough_Forest<LabelType> *
pythonHFImportFromHDF5(std::string filename)
{

    VIGRA_UNIQUE_PTR<Hough_Forest<LabelType> > hf(
            new Hough_Forest<LabelType>);

    vigra_precondition(hf_import_HDF5(*hf,filename),
            "HoughForest(): Unable to load from HDF5 file.");

    return hf.release();
}
#endif // HasHDF5

template<class LabelType>
Hough_Forest<LabelType>*
pythonHFCostruct(int treeCount, int mtry, int min_split_node_size,
        int training_set_size, float training_set_proportions,
        bool sample_with_replacement, bool sample_classes_individually,
        int max_depth)

{
    RandomForestOptions options;
    options .sample_with_replacement(sample_with_replacement) .tree_count(
            treeCount);
    //FIXME this is a conceptual error by Raul should be indeed member of the options
    //.min_split_node_size(min_split_node_size);

    if (mtry > 0)
        options.features_per_node(mtry);

    if (training_set_size != 0)
        options.samples_per_tree(training_set_size);
    else
        options.samples_per_tree(training_set_proportions);

    if (sample_classes_individually)
        options.use_stratification(RF_EQUAL);

    Hough_Forest<LabelType>* Hf = new Hough_Forest<LabelType> (options,
            max_depth, min_split_node_size);
    return Hf;

}

template<class LabelType, class FeatureType>
void pythonHFReLearnTree(Hough_Forest<LabelType> & Hf, NumpyArray<2,
        FeatureType> trainData, NumpyArray<2, LabelType> trainLabels,
        int treeId)
{
    Hf.reLearnTree(trainData, trainLabels, treeId);
}

template<class LabelType, class FeatureType>
void pythonHFLearn(Hough_Forest<LabelType>& Hf,
        NumpyArray<2, FeatureType> trainData,
        NumpyArray<2, LabelType> trainLabels, std::string split_criteria)
{
    std::cout << "The chosen split criteria is: " << split_criteria
        << std::endl;

    {
        PyAllowThreads _pythread;

        if (split_criteria == "random entropy")
            Hf.learnRandomEntropy(trainData, trainLabels);
        else if (split_criteria == "orthogonal entropy")
            Hf.learnOrthogonalEntropy(trainData, trainLabels);
        else if (split_criteria == "orthogonal gini")
            Hf.learnOrthogonalGini(trainData, trainLabels);
        else
            Hf.learnRandomGini(trainData, trainLabels);
    }
}

template<class LabelType, class FeatureType>
NumpyAnyArray pythonHFPredict(Hough_Forest<LabelType> & Hf,
        NumpyArray<2, FeatureType> testData,
        NumpyArray<2, LabelType> res)
{
    //construct result
    res.reshapeIfEmpty(MultiArrayShape<2>::type(testData.shape(0), 2),
            "Output array has wrong dimensions.");
    {
        PyAllowThreads _pythread;
        Hf.predict(testData, res);
    }
    return res;
}

//This method givent the patches arranged in a certain matrix
//create the HOUGH IMAGE
template<class LabelType, class FeatureType>
NumpyAnyArray pythonHFPredictOnImage(Hough_Forest<LabelType> & Hf,
        NumpyArray< 2, FeatureType> testData,
        NumpyArray<2, FeatureType> patchCenters,
        int imgwidth, int imgheight, int factor,
        NumpyArray<2, LabelType> res)
{
    res.reshapeIfEmpty(MultiArrayShape<2>::type(imgheight, imgwidth),
            "Output array has wrong dimensions.");

    //res.reshape(MultiArrayShape<2>::type(imgheight, imgwidth));
    {
        PyAllowThreads _pythread;
        Hf.predictOnImage(testData, patchCenters, imgwidth, imgheight, factor,
                res);
    }
    return res;
}


//This method givent the patches arranged in a certain matrix
//create the HOUGH IMAGE
template<class LabelType, class FeatureType>
NumpyAnyArray pythonHFPredictOnImageWithAngle(Hough_Forest<LabelType> & Hf,
        NumpyArray< 2, FeatureType> testData,
        NumpyArray<2, FeatureType> patchCenters,
        int imgwidth, int imgheight, int bins, int factor,
        NumpyArray<3, LabelType> res)
{
    res.reshapeIfEmpty(MultiArrayShape<3>::type(imgheight, imgwidth,bins),
            "Output array has wrong dimensions.");

    //res.reshape(MultiArrayShape<2>::type(imgheight, imgwidth));
    {
        PyAllowThreads _pythread;
        Hf.predictOnImageWithAngle(testData, patchCenters, imgwidth, imgheight,
                bins,factor, res);
    }
    return res;
}


void defineHoughForest()
{
using namespace python;
using namespace vigra;

docstring_options doc_options(true, true, false);

//This is the real functional class exposed to python
class_<Hough_Forest<float> > PythonHF("HoughForest", python::no_init);

PythonHF.def(
        "__init__",
        python::make_constructor(registerConverters(
                &pythonHFCostruct<float> ),
            boost::python::default_call_policies(),
            (arg("treeCount") = 10, arg("mtry") = -1,
             arg("min_split_node_size") = 20, arg("training_set_size") = 0,
             arg("training_set_proportions") = 1.0,
             arg("sample_with_replacement") = true,
             arg("sample_classes_individually") = true, arg("max_depth") = 16)),

        "Constructor::\n\n"
        "  Hough_Forest(treeCount = 10, mtry=RF_SQRT, min_split_node_size=1,\n"
        "               training_set_size=0, training_set_proportions=1.0,\n"
        "               sample_with_replacement=True, sample_classes_individually=False,\n"
        "               prepare_online_learning=False)\n\n"
        "'treeCount' controls the number of trees that are created.\n\n"
        "See RandomForest_ and RandomForestOptions_ in the C++ documentation "
        "for the meaning of the other parameters.\n");

#ifdef HasHDF5
PythonHF.def("__init__",python::make_constructor(
            registerConverters (&pythonHFImportFromHDF5<float>),
            boost::python::default_call_policies(),
            ( arg("filename"))),
        "Load from HDF5 file::\n\n"
        "  RandomForest(filename, pathInFile)\n\n");


PythonHF.def("writeHDF5",
        (void (*)(const Hough_Forest<float, StridedArrayTag> &, std::string const &))&hf_saveToHDF5,
        (arg("filename")),
        "Store the random forest in the given HDF5 file 'filname' under the internal\n"
        "path 'pathInFile'. If a dataset already exists, 'overwriteflag' determines\n"
        "if the old data are overwritten.\n");
#endif // HasHDF5

/*
   PythonHF.def("featureCount",
   &Hough_Forest<float>::column_count,
   "Returns the number of features the RandomForest works with.\n");
   PythonHF.def("labelCount",
   &Hough_Forest<float>::class_count,
   "Returns the number of labels, the RandomForest knows.\n");
   PythonHF.def("treeCount",
   &Hough_Forest<float>::tree_count,
   "Returns the 'treeCount', that was set when constructing the RandomForest.\n");
   */

/*
   PythonHF.def("getTrainLabels", registerConverters(&pythonHFgetTrainLabels<
   float> ), (arg("res") = python::object()), " ");
   */
PythonHF.def("predict",
        registerConverters(&pythonHFPredict<float, float> ),
        (arg("testData"), arg("res") = python::object()),
        "Predict labels on 'testData'.\n\n"
        "The output is the probability for the patch "
        "to be background or foreground for the sample.\n");

PythonHF.def("learn", registerConverters(&pythonHFLearn<float, float> ),
        (arg("trainData"), arg("trainLabels"),
         arg("splitMode") = "random gini"), "  ");
/*
   PythonHF.def("predictMean",
   registerConverters(&pythonHFPredictMean<float,float>),
   (arg("testData"),arg("res")=python::object()));

   PythonHF.def("predictMeanThroughVariance",
   registerConverters(&pythonHFPredictMeanThroughVariance<float,float>),
   (arg("testData"),
   arg("res")=python::object())
   );
   */
PythonHF.def("predictOnImage",
        registerConverters(&pythonHFPredictOnImage< float, float> ),
        (arg("testData"), arg("patchCenters"), arg("imagewidth"),
         arg("imgheigth"), arg("factor")=30, arg("res") = object()));


/*
   PythonHF.def("predictOnImageWithAngle", registerConverters(&pythonHFPredictOnImageWithAngle<
   float, float> ), (arg("testData"), arg("patchCenters"), arg(
   "imagewidth"), arg("imgheigth"),arg("bins")=10, arg("factor")=30, arg("res") = python::object()));
   */
}
}

