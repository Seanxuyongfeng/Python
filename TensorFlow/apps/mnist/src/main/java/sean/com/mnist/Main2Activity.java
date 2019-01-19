package sean.com.mnist;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.PixelFormat;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.NinePatchDrawable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main2Activity extends AppCompatActivity {
    private static final String TAG = "Main2Activity";
    private ListView mListView;
    private TextView mTextView;
    private Button mButton;
    private ImageClassifier2 classifier;
    private List<Integer> mImageRes = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        mImageRes.add(R.drawable.a00000);
        mImageRes.add(R.drawable.a00001);
        mImageRes.add(R.drawable.a00002);
        mImageRes.add(R.drawable.a00003);
        mImageRes.add(R.drawable.a00004);
        mImageRes.add(R.drawable.a00005);
        mImageRes.add(R.drawable.a00006);
        mImageRes.add(R.drawable.a00007);
        mImageRes.add(R.drawable.a00008);
        mImageRes.add(R.drawable.a00009);
        mImageRes.add(R.drawable.a00010);
        mImageRes.add(R.drawable.a00011);
        mListView = (ListView)findViewById(R.id.listView);
        mListView.setAdapter(new ListViewAdapter(this, mImageRes));
        mListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Bitmap bitmap = getBitmap(mImageRes.get(position));
                classifyFrame(bitmap);
            }
        });

        mTextView = (TextView)findViewById(R.id.textView);
        mTextView.setText("Result");
        try {
            classifier = new ImageClassifier2(this);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.");
        }

    }
    Bitmap drawable2Bitmap(Drawable drawable) {
        if (drawable instanceof BitmapDrawable) {
            return ((BitmapDrawable) drawable).getBitmap();
        } else if (drawable instanceof NinePatchDrawable) {
            Bitmap bitmap = Bitmap
                    .createBitmap(
                            drawable.getIntrinsicWidth(),
                            drawable.getIntrinsicHeight(),
                            drawable.getOpacity() != PixelFormat.OPAQUE ? Bitmap.Config.ARGB_8888
                                    : Bitmap.Config.RGB_565);
            Canvas canvas = new Canvas(bitmap);
            drawable.setBounds(0, 0, drawable.getIntrinsicWidth(),
                    drawable.getIntrinsicHeight());
            drawable.draw(canvas);
            return bitmap;
        } else {
            return null;
        }
    }
    private void classifyFrame(Bitmap bitmap) {
        //Drawable drawable = view.getDrawable();
        //Bitmap bitmap = drawable2Bitmap(drawable);

        if (classifier == null) {
            showToast("Uninitialized Classifier or invalid context.");
            return;
        }

        //}
        String textToShow = classifier.classifyFrame(bitmap);
        bitmap.recycle();
        showToast(textToShow);
    }

    private void showToast(final String text) {
        mTextView.setText(text);
    }

    private Bitmap getBitmap(int resId){
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inJustDecodeBounds = true;
        BitmapFactory.decodeResource(getResources(), resId, opts);
        opts.inSampleSize = 1;
        opts.inScaled = false;
        opts.inJustDecodeBounds = false;
        Bitmap bitmap =BitmapFactory.decodeResource(getResources(), resId, opts);
        return bitmap;
    }

    @Override
    public void onDestroy() {
        classifier.close();
        super.onDestroy();
    }
}
