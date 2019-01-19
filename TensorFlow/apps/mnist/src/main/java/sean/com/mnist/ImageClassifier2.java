package sean.com.mnist;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class ImageClassifier2 {
    private static final String TAG = "ImageClassifier2";
    private static final String MODEL_PATH = "mnist_frozen_graph.lite";

    static final int DIM_IMG_SIZE_X = 28;
    static final int DIM_IMG_SIZE_Y = 28;
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    private Interpreter tflite;
    private ByteBuffer imgData = null;
    private float[][] labelProbArray = null;
    public ImageClassifier2(Context context) throws IOException {
        tflite = new Interpreter(loadModelFile(context));

        imgData =
                ByteBuffer.allocateDirect(
                        4  * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y );
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new float[1][10];

        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    private MappedByteBuffer loadModelFile(Context activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    String classifyFrame(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            return "Uninitialized Classifier.";
        }
        convertBitmapToByteBuffer(bitmap);
        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        tflite.run(imgData, labelProbArray);
        float tmp = 0.0f;
        int index = -1;
        float sum = 0;
        for(int i = 0; i<10;i++){
            sum += labelProbArray[0][i];
            if(tmp < labelProbArray[0][i]){
                index = i;
                tmp = labelProbArray[0][i];
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));
        String textToShow = "number:"+(index);;
        textToShow = "takes "+Long.toString(endTime - startTime) + "ms to specify " + textToShow;
        return textToShow;
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                int alpha = val&0xFF000000;
                int red = ((val  & 0x00FF0000 ) >> 16);
                int green = ((val & 0x0000FF00) >> 8);
                int blue = (val & 0x000000FF);
                int grey = (int)((float) red * 0.3 + (float)green * 0.59 + (float)blue * 0.11);
                float new_grey = ((grey << 16) | (grey << 8) | grey)/255.0f;
                if(new_grey != 0.0){
                    float newa = new_grey;
                }
                imgData.putFloat(new_grey);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }
    public void close() {
        tflite.close();
        tflite = null;
    }
}
