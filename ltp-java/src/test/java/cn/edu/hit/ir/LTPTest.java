package cn.edu.hit.ir;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;


public class LTPTest {
    @Test
    public void test() throws Exception {
        LTP ltp = new LTP("../models/small", 1, 0);
        ArrayList<String> inputs = new ArrayList<>();
        inputs.add("他叫汤姆去拿外衣！");
        List output = ltp.pipeline(inputs);
        System.out.println(output);
    }

}
