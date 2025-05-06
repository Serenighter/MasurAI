package lakesapi.com.common;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PythonRunner {
    private final String pythonPath;
    private Process process;

    public PythonRunner(String pythonPath) {
        this.pythonPath = pythonPath;
    }

    /** Startuje skrypt, ale nie czeka na zakończenie. */
    public Process start(String scriptPath, String... args) throws IOException {
        List<String> cmd = new ArrayList<>();
        cmd.add(pythonPath);
        cmd.add(scriptPath);
        Collections.addAll(cmd, args);

        process = new ProcessBuilder(cmd)
                .redirectErrorStream(true)
                .start();

        // opcjonalnie: w osobnym wątku odczytujemy stdout
        new Thread(() -> {
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                br.lines().forEach(System.out::println);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();

        return process;
    }

    /** Czeka aż uruchomiony proces Pythona się zakończy i zwraca kod wyjścia. */
    public int waitForFinish() throws InterruptedException {
        if (process == null) {
            throw new IllegalStateException("Najpierw wywołaj start(...)");
        }
        return process.waitFor();
    }
}
