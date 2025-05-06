package lakesapi.com;

import lakesapi.com.common.PythonRunner;

import java.io.IOException;

public class Main {
  public static void main(String[] args) throws InterruptedException, IOException {
    PythonRunner runner = new PythonRunner("src/main/resources/analyze/.venv/Scripts/python.exe");
    runner.start("src/main/resources/analyze/image_analyze.py");
    int code = runner.waitForFinish();
    System.out.println("Exit code = " + code);


  }
}
