# Quickstart

This is a basic Java example that sets up a PostgreSQL database with Lantern, and uses the `java.sql` library to interact with the database and perform vector searches.

## Code

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Arrays;

public class LanternQuickstart {
    public static void main(String[] args) {
        try {
            // Connect to the database
            Connection conn = DriverManager.getConnection(
                    "jdbc:postgresql://localhost:5432/ourdb", "postgres", "postgres");

            // Creating a Table
            Statement stmt = conn.createStatement();
            String createTableQuery = "CREATE TABLE small_world (id integer, vector real[]);";
            stmt.execute(createTableQuery);

            // Inserting Data
            PreparedStatement pstmt = conn.prepareStatement("INSERT INTO small_world (id, vector) VALUES (?, ?);");
            pstmt.setInt(1, 0);
            pstmt.setArray(2, conn.createArrayOf("float", new Float[]{0f, 0f, 0f}));
            pstmt.executeUpdate();

            Float[][] vectors = {
                    {0f, 0f, 1f},
                    {0f, 1f, 1f},
                    {1f, 1f, 1f},
                    {2f, 0f, 1f}
            };

            for (int i = 1; i <= vectors.length; i++) {
                pstmt.setInt(1, i);
                pstmt.setArray(2, conn.createArrayOf("float", vectors[i - 1]));
                pstmt.executeUpdate();
            }

            // Creating an Index
            String createIndexQuery = "CREATE INDEX ON small_world USING hnsw (vector);";
            stmt.execute(createIndexQuery);

            // Vector Search
            stmt.execute("SET enable_seqscan = false;");
            ResultSet rs = stmt.executeQuery("SELECT id, l2sq_dist(vector, ARRAY[0,0,0]) AS dist, vector FROM small_world ORDER BY vector <-> ARRAY[0,0,0] LIMIT 3;");

            while (rs.next()) {
                int id = rs.getInt("id");
                double dist = rs.getDouble("dist");
                Float[] vector = (Float[]) rs.getArray("vector").getArray();
                System.out.printf("Vector %s with ID %d has a L2-squared distance of %.2f from [0,0,0]%n", Arrays.toString(vector), id, dist);
            }

            // Close the connection
            conn.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```
