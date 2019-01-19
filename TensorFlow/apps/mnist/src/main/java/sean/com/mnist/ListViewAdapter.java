package sean.com.mnist;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;

import java.util.ArrayList;
import java.util.List;

public class ListViewAdapter extends BaseAdapter {
    private LayoutInflater mInflater;
    List<View> mResourceId = new ArrayList<>();
    private Context mContext;

    public ListViewAdapter(Context context, List<Integer> list) {
        mInflater = LayoutInflater.from(context);
        mContext =context;
        for(int res : list){
            addImage(res);
        }
    }

    private void addImage(int resId){
        ImageView image1 = new ImageView(mContext);
        ViewGroup.LayoutParams params = image1.getLayoutParams();
        image1.setImageResource(resId);
        mResourceId.add(image1);
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        if(convertView == null){
            return mResourceId.get(position);
        }
        return convertView;
    }

    @Override
    public int getCount() {
        return mResourceId.size();
    }

    @Override
    public Object getItem(int position) {
        return null;
    }

    @Override
    public long getItemId(int position) {
        return position;
    }


}
